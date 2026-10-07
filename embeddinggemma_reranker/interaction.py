"""Optional token-level query/document interaction.

A bi-encoder throws away token alignment: it reduces each side to one vector and
then compares them. This module keeps token granularity and lets document tokens
attend to query tokens, which is what lets a reranker learn things like "this
document mentions JWT *and* expiration" that a dot product cannot express.

Memory note: cross-attention costs ``O(T_q * T_d)``. With a 270M backbone that is
cheap at the default budgets (64 query / 256 document tokens) and expensive at long
lengths, so token budgets are explicit config and the module is off by default.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InteractionConfig


class TokenInteraction(nn.Module):
    """Cross-attention plus an interaction-matrix aggregator.

    Pipeline::

        query tokens      (B, Tq, H)
        document tokens   (B, Td, H)
              |
              v
        project both to ``projection_dim``
              |
              v
        document attends to query  ->  context (B, Td, P)
              |
              v
        elementwise interaction map (B, Td, Tq) from projected q/d
              |
              v
        aggregate over the query axis (mean / max) and the document axis
              |
              v
        output vector (B, projection_dim)
    """

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_max_pool: bool = True,
        normalize_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.use_max_pool = use_max_pool
        self.normalize_tokens = normalize_tokens

        self.query_proj = nn.Linear(hidden_size, projection_dim)
        self.document_proj = nn.Linear(hidden_size, projection_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_norm = nn.LayerNorm(projection_dim)
        self.document_norm = nn.LayerNorm(projection_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(projection_dim + (2 if use_max_pool else 1), projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @classmethod
    def from_config(
        cls, hidden_size: int, config: InteractionConfig
    ) -> "TokenInteraction":
        return cls(
            hidden_size=hidden_size,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_max_pool=config.use_max_pool,
            normalize_tokens=config.normalize_tokens,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        document_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        document_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the token-interaction vector for each pair.

        Parameters
        ----------
        query_tokens: ``(B, Tq, H)``
        document_tokens: ``(B, Td, H)``
        query_mask / document_mask: optional ``(B, T)`` boolean masks.
        """
        if query_tokens.dim() != 3 or document_tokens.dim() != 3:
            raise ValueError(
                "token interaction expects 3-D (B, T, H) inputs, got "
                f"{tuple(query_tokens.shape)} and {tuple(document_tokens.shape)}"
            )
        if query_tokens.shape[0] != document_tokens.shape[0]:
            raise ValueError("query and document batches must match")

        q = self.query_norm(self.query_proj(query_tokens))
        d = self.document_norm(self.document_proj(document_tokens))

        if self.normalize_tokens:
            q = F.normalize(q, p=2.0, dim=-1)
            d = F.normalize(d, p=2.0, dim=-1)

        query_mask = torch.ones(q.shape[:2], device=q.device, dtype=torch.bool) if query_mask is None else query_mask.bool()
        document_mask = torch.ones(d.shape[:2], device=d.device, dtype=torch.bool) if document_mask is None else document_mask.bool()
        if not query_mask.any(dim=-1).all() or not document_mask.any(dim=-1).all():
            raise ValueError("every sequence must contain an unmasked token")
        key_padding_mask = ~query_mask

        context, _attn = self.attention(
            d, q, q, key_padding_mask=key_padding_mask, need_weights=False
        )

        # Interaction map: every document token against every query token.
        interaction = torch.einsum("btd,bqd->btq", d, q)

        valid = document_mask.unsqueeze(2) & query_mask.unsqueeze(1)
        interaction = interaction.masked_fill(~valid, 0.0)
        context_mean = (context * document_mask.unsqueeze(-1)).sum(dim=1) / document_mask.sum(dim=1, keepdim=True)
        interaction_mean = interaction.sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
        pooled = [context_mean, interaction_mean.unsqueeze(-1)]
        if self.use_max_pool:
            maxima = interaction.masked_fill(~query_mask.unsqueeze(1), torch.finfo(interaction.dtype).min).max(dim=2).values
            pooled.append(((maxima * document_mask).sum(dim=1) / document_mask.sum(dim=1)).unsqueeze(-1))

        combined = torch.cat(pooled, dim=-1)
        output = self.output_proj(combined)

        return {
            "token_context": output,
            "interaction_matrix": interaction,
        }


class LateInteractionScorer(nn.Module):
    """MaxSim-style late interaction over token embeddings.

    Kept separate from :class:`TokenInteraction` because it produces a *score*
    rather than a feature vector, and some deployments want the cheap
    ``max_sim(q_i, d_j)`` aggregation without any cross-attention parameters.
    """

    def __init__(self, hidden_size: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, projection_dim, bias=False)
        self.document_proj = self.query_proj

    @classmethod
    def from_config(
        cls, hidden_size: int, config: InteractionConfig
    ) -> "LateInteractionScorer":
        return cls(hidden_size, config.projection_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        document_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        document_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-pair MaxSim scores ``(B,)``."""
        q = F.normalize(self.query_proj(query_tokens), p=2.0, dim=-1)
        d = F.normalize(self.document_proj(document_tokens), p=2.0, dim=-1)

        scores = torch.einsum("bqd,btd->bqt", q, d)

        if document_mask is not None:
            doc_mask = document_mask.to(torch.bool)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~doc_mask.unsqueeze(1), neg_inf)
        if document_mask is not None and not document_mask.bool().any(dim=-1).all():
            raise ValueError("every document must contain an unmasked token")
        maxima = scores.max(dim=-1).values
        if query_mask is None:
            return maxima.mean(dim=-1)
        if not query_mask.bool().any(dim=-1).all():
            raise ValueError("every query must contain an unmasked token")
        return (maxima * query_mask).sum(dim=-1) / query_mask.sum(dim=-1)


def build_interaction(
    hidden_size: int, config: InteractionConfig
) -> Optional[TokenInteraction]:
    """Return the module when enabled, else ``None``."""
    return TokenInteraction.from_config(hidden_size, config) if config.enabled else None
