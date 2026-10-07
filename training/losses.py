"""Configurable ranking losses.

All losses take **raw logits** plus supervision and return a scalar. Nothing here
squashes scores into [0, 1] first, because BCE and margin ranking both need the
unbounded geometry to work.

Available:

* ``bce``             — binary cross-entropy per (query, document) pair
* ``pairwise_margin`` — ``max(0, margin - (s_pos - s_neg))`` over all valid pairs
* ``listwise``        — softmax cross-entropy over a candidate list (ListNet-style)
* ``mse``             — regression against graded relevance
* ``distill_kl``      — KL to externally supplied teacher scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from embeddinggemma_reranker.config import SUPPORTED_LOSSES, ConfigError


@dataclass
class LossOutput:
    """A loss value plus the pieces a training log wants."""

    loss: torch.Tensor
    metrics: dict


def _check_shape(scores: torch.Tensor, relevance: torch.Tensor) -> None:
    if scores.shape != relevance.shape:
        raise ValueError(
            f"scores {tuple(scores.shape)} and relevance {tuple(relevance.shape)} "
            "must have the same shape"
        )
    if scores.dim() != 2:
        raise ValueError(f"expected (B, N) scores, got {tuple(scores.shape)}")


def pairwise_margin_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.1,
) -> LossOutput:
    """Margin ranking over every (positive, negative) pair in the list.

    Uses a ``max(0, ...)`` formulation rather than summing, so one badly wrong
    candidate cannot dominate the gradient.
    """
    _check_shape(scores, relevance)
    batch, num = scores.shape
    device = scores.device

    if mask is None:
        mask = torch.ones_like(relevance, dtype=torch.bool)
    mask = mask.to(torch.bool)

    positives = (relevance.unsqueeze(2) > relevance.unsqueeze(1)) & mask.unsqueeze(2) & mask.unsqueeze(1)
    num_pairs = int(positives.sum().item())
    if num_pairs == 0:
        zero = scores.sum() * 0.0
        return LossOutput(
            loss=zero,
            metrics={"pairwise_margin_loss": 0.0, "num_pairs": 0, "pairs_per_query": 0.0},
        )

    diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, pos, neg)
    rel_diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)
    losses = F.relu(margin - diff) * rel_diff.clamp(min=0.0)

    weights = positives.to(scores.dtype)
    loss = (losses * weights).sum() / weights.sum().clamp(min=1.0)

    accuracy = ((diff > 0).to(scores.dtype) * weights).sum() / weights.sum().clamp(min=1.0)

    return LossOutput(
        loss=loss,
        metrics={
            "pairwise_margin_loss": float(loss.item()),
            "num_pairs": num_pairs,
            "pairs_per_query": num_pairs / max(1, batch),
            "pairwise_accuracy": float(accuracy.item()),
            "margin_violations": int(((diff <= 0) & weights.bool()).sum().item()),
        },
    )


def bce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> LossOutput:
    """Binary cross-entropy on calibrated-looking logits."""
    _check_shape(scores, labels)
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    mask = mask.to(torch.bool)

    losses = F.binary_cross_entropy_with_logits(
        scores, labels.to(scores.dtype), reduction="none"
    )
    weights = mask.to(scores.dtype)
    total = weights.sum().clamp(min=1.0)
    loss = (losses * weights).sum() / total

    return LossOutput(
        loss=loss,
        metrics={"bce_loss": float(loss.item()), "num_supervised": int(mask.sum().item())},
    )


def listwise_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> LossOutput:
    """Softmax cross-entropy between predicted and relevance-derived distributions.

    The target is a relevance-weighted distribution (higher grade ⇒ more mass).
    Padding candidates are removed before the softmax, so a short list does not
    train against phantom zeros.
    """
    _check_shape(scores, relevance)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if mask is None:
        mask = torch.ones_like(relevance, dtype=torch.bool)
    mask = mask.to(torch.bool)

    neg_inf = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, neg_inf)
    masked_relevance = relevance.masked_fill(~mask, neg_inf)

    log_p = F.log_softmax(masked_scores / temperature, dim=-1)
    target = F.softmax(masked_relevance / temperature, dim=-1)

    per_query = -(target * log_p).sum(dim=-1)
    valid = mask.any(dim=-1)
    loss = per_query[valid].mean() if bool(valid.any()) else scores.sum() * 0.0

    return LossOutput(
        loss=loss,
        metrics={
            "listwise_loss": float(loss.item()),
            "num_queries": int(valid.sum().item()),
        },
    )


def mse_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> LossOutput:
    """Mean squared error against graded relevance values.

    Note that targets are raw grades (0, 1, 2, 3...), not probabilities; scale
    your grades or this term will dominate.
    """
    _check_shape(scores, targets)
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    mask = mask.to(torch.bool)

    errors = (scores - targets.to(scores.dtype)) ** 2
    weights = mask.to(scores.dtype)
    loss = (errors * weights).sum() / weights.sum().clamp(min=1.0)

    return LossOutput(
        loss=loss,
        metrics={"mse_loss": float(loss.item()), "num_supervised": int(mask.sum().item())},
    )


def distillation_kl_loss(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> LossOutput:
    """KL divergence to teacher score distributions.

    Teacher scores are supplied externally (see ``teacher_scores`` in the JSONL
    schema). Nothing here invents a teacher: if the field is missing the caller
    must skip this term.
    """
    _check_shape(scores, teacher_scores)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    mask = mask.to(torch.bool)

    neg_inf = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, neg_inf) / temperature
    masked_teacher = teacher_scores.masked_fill(~mask, neg_inf) / temperature

    log_p = F.log_softmax(masked_scores, dim=-1)
    target = F.softmax(masked_teacher, dim=-1)

    per_query = F.kl_div(log_p, target, reduction="none").sum(dim=-1)
    valid = mask.any(dim=-1)
    loss = per_query[valid].mean() if bool(valid.any()) else scores.sum() * 0.0

    return LossOutput(
        loss=loss,
        metrics={"distill_kl_loss": float(loss.item()), "num_queries": int(valid.sum().item())},
    )


def compute_loss(
    name: str,
    scores: torch.Tensor,
    batch,
    margin: float = 0.1,
    listwise_temperature: float = 1.0,
    distillation_weight: float = 0.0,
) -> LossOutput:
    """Dispatch to the configured loss and optionally add a distillation term.

    Parameters
    ----------
    name: one of :data:`embeddinggemma_reranker.config.SUPPORTED_LOSSES`.
    scores: ``(B, N)`` raw logits from the relevance head.
    batch: a :class:`~training.collator.RerankerBatch`.
    """
    if name not in SUPPORTED_LOSSES:
        raise ConfigError(f"unknown loss {name!r}; expected one of {SUPPORTED_LOSSES}")

    mask = getattr(batch, "pair_mask", None)
    if mask is None:
        mask = getattr(batch, "document_mask", None)

    if name == "bce":
        output = bce_loss(scores, batch.binary_labels, mask=mask)
    elif name == "pairwise_margin":
        output = pairwise_margin_loss(
            scores, batch.relevance, mask=mask, margin=margin
        )
    elif name == "listwise":
        output = listwise_loss(
            scores, batch.relevance, mask=mask, temperature=listwise_temperature
        )
    elif name == "mse":
        output = mse_loss(scores, batch.relevance, mask=mask)
    else:  # distill_kl
        if batch.teacher_scores is None:
            raise ConfigError(
                "distill_kl requires 'teacher_scores' in the data; refusing to "
                "fabricate teacher labels"
            )
        output = distillation_kl_loss(
            scores, batch.teacher_scores, mask=mask, temperature=listwise_temperature
        )

    total = output.loss
    metrics = dict(output.metrics)

    if distillation_weight > 0 and name != "distill_kl":
        if batch.teacher_scores is None:
            metrics["distill_skipped"] = "no teacher_scores in batch"
        else:
            distill = distillation_kl_loss(
                scores, batch.teacher_scores, mask=mask, temperature=listwise_temperature
            )
            total = total + distillation_weight * distill.loss
            metrics["distill_kl_loss"] = distill.metrics["distill_kl_loss"]
            metrics["distill_weight"] = distillation_weight

    return LossOutput(loss=total, metrics=metrics)


def available_losses() -> tuple[str, ...]:
    return SUPPORTED_LOSSES
