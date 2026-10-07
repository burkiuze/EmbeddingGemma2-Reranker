"""Tests for token-level interaction and late interaction."""

from __future__ import annotations

import pytest
import torch

from embeddinggemma_reranker.config import InteractionConfig
from embeddinggemma_reranker.interaction import (
    LateInteractionScorer,
    TokenInteraction,
    build_interaction,
)


def test_token_interaction_output_shape():
    module = TokenInteraction(hidden_size=16, projection_dim=32, num_heads=4)
    query = torch.randn(3, 5, 16)
    document = torch.randn(3, 9, 16)

    out = module(query, document)

    assert out["token_context"].shape == (3, 32)
    assert out["interaction_matrix"].shape == (3, 9, 5)


def test_interaction_matrix_is_query_by_document_tokens():
    module = TokenInteraction(hidden_size=4, projection_dim=4, num_heads=1)
    query = torch.randn(1, 2, 4)
    document = torch.randn(1, 3, 4)

    matrix = module(query, document)["interaction_matrix"]

    # (B, Td, Tq): document tokens on axis 1, query tokens on axis 2.
    assert matrix.shape == (1, 3, 2)


def test_token_interaction_respects_masks():
    module = TokenInteraction(hidden_size=8, projection_dim=8, num_heads=2)
    query = torch.randn(2, 4, 8)
    document = torch.randn(2, 6, 8)

    query_mask = torch.tensor([[True, True, False, False]] * 2)
    document_mask = torch.tensor([[True, True, True, False, False, False]] * 2)

    out = module(query, document, query_mask=query_mask, document_mask=document_mask)

    assert torch.isfinite(out["token_context"]).all()
    assert out["interaction_matrix"].shape == (2, 6, 4)


def test_token_interaction_is_differentiable():
    module = TokenInteraction(hidden_size=8, projection_dim=8, num_heads=2)
    query = torch.randn(2, 3, 8, requires_grad=True)
    document = torch.randn(2, 4, 8, requires_grad=True)

    module(query, document)["token_context"].sum().backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert document.grad is not None and torch.isfinite(document.grad).all()


def test_token_interaction_rejects_2d_inputs():
    module = TokenInteraction(hidden_size=8, projection_dim=8, num_heads=2)
    with pytest.raises(ValueError, match="3-D"):
        module(torch.randn(2, 8), torch.randn(2, 8))


def test_token_interaction_rejects_batch_mismatch():
    module = TokenInteraction(hidden_size=8, projection_dim=8, num_heads=2)
    with pytest.raises(ValueError, match="batches must match"):
        module(torch.randn(2, 3, 8), torch.randn(3, 3, 8))


def test_build_interaction_returns_none_when_disabled():
    assert build_interaction(16, InteractionConfig(enabled=False)) is None


def test_build_interaction_returns_module_when_enabled():
    module = build_interaction(16, InteractionConfig(enabled=True, projection_dim=24))
    assert isinstance(module, TokenInteraction)


def test_late_interaction_scores_shape():
    scorer = LateInteractionScorer(hidden_size=16, projection_dim=32)
    out = scorer(torch.randn(3, 4, 16), torch.randn(3, 7, 16))
    assert out.shape == (3,)


def test_late_interaction_identical_inputs_score_high():
    """A perfect query/document match should beat an unrelated one.

    Uses untrained random projections, so this only asserts the *mechanism*
    (self-similarity beats cross-similarity), not any semantic quality.
    """
    torch.manual_seed(0)
    scorer = LateInteractionScorer(hidden_size=32, projection_dim=32)

    query = torch.randn(1, 4, 32)
    match = scorer(query, query).item()
    unrelated = scorer(query, torch.randn(1, 6, 32)).item()

    assert match > unrelated


def test_late_interaction_single_token_document():
    scorer = LateInteractionScorer(hidden_size=8, projection_dim=8)
    out = scorer(torch.randn(2, 3, 8), torch.randn(2, 1, 8))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_from_config_respects_projection_dim():
    module = TokenInteraction.from_config(
        hidden_size=12, config=InteractionConfig(projection_dim=40, num_heads=4)
    )
    out = module(torch.randn(2, 3, 12), torch.randn(2, 5, 12))
    assert out["token_context"].shape == (2, 40)
