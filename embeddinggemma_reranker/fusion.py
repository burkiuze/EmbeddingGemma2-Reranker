"""Query/document feature fusion.

The lightweight default (``elementwise``) builds the classic four-feature set

    Q,  D,  Q * D,  |Q - D|

and pushes it through a learned projection. The mechanism is configurable so a
deeper interaction can be added without touching the reranker layers.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig


def _activations() -> Dict[str, type]:
    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class QueryDocumentFusion(nn.Module):
    """Combine query and document representations into a ranking vector.

    Parameters
    ----------
    query_dim, document_dim:
        Widths of the representations handed in. They are allowed to differ —
        the reranker projects both to a common width first.
    output_dim:
        Width of the produced ranking representation.
    interaction_type:
        One of ``none``, ``concatenate``, ``elementwise``, ``dot``, ``token``.
        ``token`` is handled by
        :class:`~embeddinggemma_reranker.interaction.TokenInteraction`; this
        module still produces the fused sequence-level vector for it.
    """

    def __init__(
        self,
        query_dim: int,
        document_dim: int,
        output_dim: int,
        interaction_type: str = "elementwise",
        dropout: float = 0.1,
        layer_norm_features: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.interaction_type = interaction_type
        self.output_dim = output_dim
        self.layer_norm_features = layer_norm_features

        self.query_proj = nn.Linear(query_dim, output_dim)
        self.document_proj = nn.Linear(document_dim, output_dim)

        multiplier = self._feature_multiplier()
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * output_dim + 1 if interaction_type == "dot" else output_dim * multiplier, output_dim),
            _activations().get(activation, nn.GELU)(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.feature_norm = nn.LayerNorm(output_dim) if layer_norm_features else nn.Identity()

    def _feature_multiplier(self) -> int:
        if self.interaction_type == "elementwise":
            return 4
        if self.interaction_type in ("concatenate", "token"):
            return 2
        if self.interaction_type in ("none", "dot"):
            return 1
        raise ValueError(f"unsupported interaction_type {self.interaction_type!r}")

    def forward(
        self, query: torch.Tensor, document: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Fuse a batch of query/document pairs.

        Parameters
        ----------
        query, document:
            ``(B, Dq)`` and ``(B, Dd)`` sequence-level representations.

        Returns
        -------
        dict with ``fused`` ``(B, output_dim)``, plus the individual pieces so
        tests and analysis can inspect what the head actually saw.
        """
        if query.dim() != 2 or document.dim() != 2:
            raise ValueError(
                f"expected 2-D query/document, got {tuple(query.shape)} and "
                f"{tuple(document.shape)}"
            )
        if query.shape[0] != document.shape[0]:
            raise ValueError(
                f"batch mismatch: query {query.shape[0]} vs document {document.shape[0]}"
            )

        q = self.query_proj(query)
        d = self.document_proj(document)

        if self.interaction_type == "elementwise":
            features = torch.cat([q, d, q * d, (q - d).abs()], dim=-1)
        elif self.interaction_type in ("concatenate", "token"):
            features = torch.cat([q, d], dim=-1)
        elif self.interaction_type == "none":
            features = q + d
        elif self.interaction_type == "dot":
            # Keep the per-feature geometry: cosine plus the two operands.
            cosine = F.cosine_similarity(q, d, dim=-1, eps=1e-8).unsqueeze(-1)
            features = torch.cat([cosine, q, d], dim=-1)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported interaction_type {self.interaction_type!r}")

        fused = self.fusion_proj(features)
        fused = self.feature_norm(fused)

        return {
            "fused": fused,
            "query_projected": q,
            "document_projected": d,
        }

    @classmethod
    def from_config(
        cls, query_dim: int, document_dim: int, config: FusionConfig
    ) -> "QueryDocumentFusion":
        return cls(
            query_dim=query_dim,
            document_dim=document_dim,
            output_dim=config.fusion_dim,
            interaction_type=config.interaction_type,
            dropout=config.dropout,
            layer_norm_features=config.layer_norm_features,
            activation=config.activation,
        )


def elementwise_features(
    query: torch.Tensor, document: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """The raw four-feature set, without projection.

    Exposed separately so tests can assert the interaction arithmetic directly
    instead of inferring it through a learned layer.
    """
    return {
        "query": query,
        "document": document,
        "product": query * document,
        "absolute_difference": (query - document).abs(),
    }
