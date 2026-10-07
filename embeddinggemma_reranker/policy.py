"""Ranking policy and abstention.

The neural network produces scores. *What to do* about those scores — when to
declare that no candidate is good enough — is policy, and lives here, outside the
forward pass. That separation matters: thresholds must be tunable per deployment
without retraining, and they must never leak into the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import PolicyConfig


@dataclass
class RankingDecision:
    """Outcome of applying policy to one candidate list."""

    accepted: bool
    reason: Optional[str] = None
    violated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"accepted": self.accepted}
        if self.reason:
            out["reason"] = self.reason
        if self.violated:
            out["violated"] = self.violated
        return out


class RankingPolicy:
    """Applies ``minimum_relevance`` / ``minimum_margin`` / ``minimum_confidence``.

    All three thresholds are optional. With none configured the policy always
    accepts and abstention never triggers — that is the default, because claiming
    "nothing is relevant" out of the box would be worse than useless.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = (config or PolicyConfig()).validate()

    @property
    def is_noop(self) -> bool:
        return (
            self.config.minimum_relevance is None
            and self.config.minimum_margin is None
            and self.config.minimum_confidence is None
        )

    def evaluate(
        self,
        top_score: Optional[torch.Tensor] = None,
        margin: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> RankingDecision:
        """Decide whether the top candidate is acceptable.

        Parameters
        ----------
        top_score: ``(B,)`` or scalar raw score of the best candidate.
        margin: ``(B,)`` or scalar top1 - top2 gap. Skipped when only one
            candidate was scored, because the margin is undefined.
        confidence: ``(B,)`` or scalar calibrated confidence in [0, 1].
        """
        if self.is_noop:
            return RankingDecision(accepted=True)

        for value, name in (
            (top_score, "minimum_relevance"),
            (margin, "minimum_margin"),
            (confidence, "minimum_confidence"),
        ):
            threshold = getattr(self.config, name)
            if threshold is None:
                continue
            if value is None:
                # A threshold was configured but the caller supplied no signal.
                # Treat it as unmet rather than silently accepting.
                return RankingDecision(
                    accepted=False,
                    reason=f"{name} could not be evaluated (missing signal)",
                    violated=name,
                )
            if bool((value < threshold).any()):
                return RankingDecision(
                    accepted=False,
                    reason=f"{name} not met",
                    violated=name,
                )
        return RankingDecision(accepted=True)


def sort_scores(
    scores: torch.Tensor, descending: bool = True
) -> torch.Tensor:
    """Return the indices that sort ``scores`` along the last dimension.

    ``torch.sort`` on CPU is stable, so equal scores keep their original order and
    ranking output is deterministic run to run.
    """
    if scores.dim() != 1:
        raise ValueError(f"expected a 1-D score vector, got {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("cannot sort an empty score vector")
    return torch.argsort(scores, descending=descending, stable=True)


def build_ranking(
    scores: torch.Tensor,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Turn one score vector into a ranked result payload.

    Preserves the caller's original indices (or document ids) alongside the raw
    score and the 1-based rank, so downstream consumers never have to guess which
    input produced which score.
    """
    if scores.dim() != 1:
        raise ValueError(f"expected 1-D scores, got {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("cannot build a ranking from zero scores")

    if indices is not None:
        indices = list(indices)
        if len(indices) != scores.numel():
            raise ValueError(
                f"indices length {len(indices)} does not match {scores.numel()} scores"
            )
    else:
        indices = list(range(scores.numel()))

    order = sort_scores(scores)
    ranked: List[Dict[str, Any]] = []
    for position, source in enumerate(order.tolist()):
        ranked.append(
            {
                "index": indices[source],
                "rank": position + 1,
                "score": float(scores[source].item()),
            }
        )
    return {
        "ranking": ranked,
        "order": [item["index"] for item in ranked],
        "scores": scores.detach().clone(),
    }
