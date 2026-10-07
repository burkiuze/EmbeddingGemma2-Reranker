"""Pooling strategies.

The upstream checkpoint ships ``1_Pooling/config.json`` with
``{"pooling_mode": "mean", "embedding_dimension": 768, "include_prompt": true}``.
``mean`` is therefore the default here, but the others exist so the reranker can
build richer sequence representations than a single average.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def prepare_mask(attention_mask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Return a ``(B, T)`` boolean mask matching ``hidden``.

    Accepts ``[B, T]``, ``[B, T, T]``, ``[B, 1, T, T]``, bool or int dtypes.
    Raises if a row has no supported position at all, because that silently
    produces NaNs downstream.
    """
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, 0, :, :]
    if attention_mask.dim() == 3:
        attention_mask = attention_mask.any(dim=-1)
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must reduce to 2 dims, got shape {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(attention_mask.shape)} does not match "
            f"hidden {tuple(hidden.shape[:2])}"
        )
    mask = attention_mask.to(torch.bool)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every sequence must contain at least one attended token")
    return mask


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(hidden.dtype).min
    filled = hidden.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return filled.max(dim=1).values


def last_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).clamp(min=1) - 1
    index = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(1, index).squeeze(1)


def attention_pool(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    query: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Single learned-query attention pooling (``query`` is a 1-D vector)."""
    if scale is None:
        scale = hidden.shape[-1] ** -0.5
    scores = torch.einsum("btd,d->bt", hidden, query) * scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
    return (hidden * weights).sum(dim=1)


def pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "mean",
    attention_query: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch to the requested pooling strategy.

    ``attention`` requires ``attention_query`` (a learned vector) and falls back
    to ``mean`` when it is absent, rather than failing at inference time.
    """
    mask = prepare_mask(attention_mask, hidden)
    if mode == "mean":
        return mean_pool(hidden, mask)
    if mode == "cls":
        return cls_pool(hidden, mask)
    if mode == "max":
        return max_pool(hidden, mask)
    if mode == "last":
        return last_pool(hidden, mask)
    if mode == "attention":
        if attention_query is None:
            return mean_pool(hidden, mask)
        return attention_pool(hidden, mask, attention_query)
    raise ValueError(
        f"unsupported pooling mode {mode!r}; expected one of "
        f"mean/cls/max/last/attention"
    )
