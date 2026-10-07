"""Relevance and confidence heads.

``RelevanceHead`` emits **one raw logit per (query, document) pair** and nothing
else. Squashing it into [0, 1] during training would break the ranking geometry
that margin and listwise losses depend on, so calibration is applied only at
inference time and only when the caller asks for it.

``ConfidenceHead`` is optional and never an arbitrary number: it sees the ranking
representation plus the actual score statistics of the candidate set (top score,
top1-top2 margin, spread), which is what makes abstention decisions meaningful.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import HeadConfig


class RelevanceHead(nn.Module):
    """Scalar relevance scorer.

    Input: the ranking representation (optionally concatenated with extra
    per-pair signals). Output: ``(B,)`` raw scores with no range constraint.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        extra_features: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.extra_features = extra_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + extra_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        config: HeadConfig,
        extra_features: int = 0,
    ) -> "RelevanceHead":
        return cls(
            input_dim=input_dim,
            hidden_dim=config.relevance_hidden_dim,
            dropout=config.relevance_dropout,
            extra_features=extra_features,
        )

    def forward(
        self, features: torch.Tensor, extras: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"expected (B, D), got {tuple(features.shape)}")
        if self.extra_features:
            if extras is None:
                raise ValueError(
                    f"head expects {self.extra_features} extra features, got none"
                )
            if extras.shape[-1] != self.extra_features:
                raise ValueError(
                    f"extra feature width mismatch: got {extras.shape[-1]}, "
                    f"expected {self.extra_features}"
                )
            features = torch.cat([features, extras], dim=-1)
        return self.mlp(features).squeeze(-1)


class ConfidenceHead(nn.Module):
    """Predict how much to trust the top of a ranking.

    Operates per *candidate list*, not per pair: a confidence value only means
    something relative to the other documents for the same query. Inputs are the
    per-candidate ranking representations pooled over the list plus explicit
    score-distribution statistics.
    """

    #: Number of distribution statistics fed alongside the representation.
    num_statistics = 5

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + self.num_statistics, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @classmethod
    def from_config(
        cls, input_dim: int, config: HeadConfig
    ) -> "ConfidenceHead":
        return cls(
            input_dim=input_dim,
            hidden_dim=config.confidence_hidden_dim,
            dropout=config.confidence_dropout,
        )

    @staticmethod
    def score_statistics(scores: torch.Tensor) -> torch.Tensor:
        """Distribution features computed from a candidate score vector.

        Parameters
        ----------
        scores: ``(B, N)`` raw scores for one query's candidates.

        Returns
        -------
        ``(B, 5)`` tensor of ``[top, margin, std, mean, range]``.

        A single-candidate list has an undefined top1-top2 margin; it is reported
        as 0 rather than NaN so downstream policies stay well-defined.
        """
        if scores.dim() != 2:
            raise ValueError(f"expected (B, N) scores, got {tuple(scores.shape)}")
        batch, num = scores.shape
        if num == 0:
            raise ValueError("cannot compute score statistics for an empty candidate set")

        sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
        top1 = sorted_scores[:, :1]
        if num > 1:
            margin = sorted_scores[:, :1] - sorted_scores[:, 1:2]
        else:
            margin = torch.zeros_like(top1)

        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, unbiased=False, keepdim=True)
        value_range = scores.max(dim=-1, keepdim=True).values - scores.min(
            dim=-1, keepdim=True
        ).values

        return torch.cat([top1, margin, std, mean, value_range], dim=-1)

    def forward(self, representations: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Return raw confidence logits ``(B,)``.

        Parameters
        ----------
        representations: ``(B, N, D)`` ranking representation per candidate.
        scores: ``(B, N)`` matching raw relevance scores.
        """
        if representations.dim() != 3:
            raise ValueError(
                f"expected (B, N, D) representations, got {tuple(representations.shape)}"
            )
        if scores.dim() != 2:
            raise ValueError(f"expected (B, N) scores, got {tuple(scores.shape)}")
        if representations.shape[:2] != scores.shape:
            raise ValueError(
                f"representation/score shape mismatch: {tuple(representations.shape[:2])} "
                f"vs {tuple(scores.shape)}"
            )

        batch, num, dim = representations.shape

        # Weight candidates by their (softmax-normalised) score so the head sees
        # a summary of the whole list rather than an arbitrary member.
        attention = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (representations * attention).sum(dim=1)
        if num == 1:
            pooled = representations[:, 0]

        statistics = self.score_statistics(scores)
        features = torch.cat([pooled, statistics], dim=-1)
        return self.mlp(features).squeeze(-1)


def calibrate(logits: torch.Tensor, method: str = "sigmoid") -> torch.Tensor:
    """Map raw logits to probabilities for user-facing output only.

    ``none`` returns the logits unchanged so callers can always reach the raw
    values regardless of what the config says.
    """
    if method == "none":
        return logits
    if method == "sigmoid":
        return torch.sigmoid(logits)
    raise ValueError(f"unsupported calibration method {method!r}")


def confidence_from_scores(scores: torch.Tensor, num_documents: int) -> torch.Tensor:
    """Deterministic, model-free fallback confidence.

    Used only when ``confidence_head=False``. It reports how peaked the score
    distribution is, which is weak but honest: it is not a learned confidence and
    the README says so.
    """
    if scores.numel() < 2:
        return torch.ones(scores.shape[:-1], device=scores.device, dtype=scores.dtype)
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    margin = sorted_scores[..., 0] - sorted_scores[..., 1]
    spread = scores.std(dim=-1, unbiased=False).clamp(min=1e-6)
    return torch.sigmoid(margin / spread)


def confidence_report(
    confidence: torch.Tensor,
    scores: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Bundle a confidence value with the statistics that produced it."""
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    report: Dict[str, torch.Tensor] = {"confidence": confidence}
    report["top_score"] = sorted_scores[..., 0]
    if scores.shape[-1] > 1:
        report["margin"] = sorted_scores[..., 0] - sorted_scores[..., 1]
    else:
        report["margin"] = torch.zeros_like(sorted_scores[..., 0])
    report["mean_score"] = scores.mean(dim=-1)
    report["std_score"] = scores.std(dim=-1, unbiased=False)
    return report
