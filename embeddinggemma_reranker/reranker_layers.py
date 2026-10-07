"""Trainable reranker blocks.

A small pre-norm transformer stack that operates on the fused ranking
representation. Deliberately narrow (``reranker_dim: 512``,
``num_reranker_layers: 2``) so the added parameter count stays small next to the
270M backbone — see ``docs/architecture.md`` for the measured breakdown.

Shape of each block::

    x -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> (+ x)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .config import RerankerLayerConfig


def _activation(name: str) -> nn.Module:
    table = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
    try:
        return table[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {name!r}; expected one of {sorted(table)}"
        ) from exc


class RerankerLayer(nn.Module):
    """One residual FFN block."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError(f"RerankerLayer expects (B, D), got {tuple(hidden.shape)}")
        return hidden + self.ffn(self.norm(hidden))


class RerankerStack(nn.Module):
    """A stack of :class:`RerankerLayer` with an entry projection.

    ``extra_features`` lets the caller concatenate additional per-pair signals
    (token-interaction output, score-distribution features) onto the fused
    vector before the stack sees it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        intermediate_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6,
        extra_features: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extra_features: List[int] = list(extra_features or [])

        total_extra = sum(self.extra_features)
        self.input_projection = nn.Linear(input_dim + total_extra, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            RerankerLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        config: RerankerLayerConfig,
        extra_features: Optional[Sequence[int]] = None,
    ) -> "RerankerStack":
        return cls(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            intermediate_dim=config.intermediate_dim,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            extra_features=extra_features,
        )

    def forward(
        self, features: torch.Tensor, extras: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"expected (B, D) features, got {tuple(features.shape)}")

        if self.extra_features:
            if extras is None or len(extras) != len(self.extra_features):
                raise ValueError(
                    f"stack expects {len(self.extra_features)} extra feature tensors, "
                    f"got {0 if extras is None else len(extras)}"
                )
            expected = list(self.extra_features)
            for tensor, want in zip(extras, expected):
                if tensor.shape[-1] != want:
                    raise ValueError(
                        f"extra feature width mismatch: got {tensor.shape[-1]}, "
                        f"expected {want}"
                    )
                if tensor.shape[0] != features.shape[0]:
                    raise ValueError("extra features must share the batch dimension")
            features = torch.cat([features, *extras], dim=-1)

        hidden = self.input_dropout(self.input_norm(self.input_projection(features)))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_norm(hidden)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim
