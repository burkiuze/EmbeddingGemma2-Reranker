"""Tests for QueryDocumentFusion."""

from __future__ import annotations

import pytest
import torch

from embeddinggemma_reranker.config import FusionConfig
from embeddinggemma_reranker.fusion import QueryDocumentFusion, elementwise_features


def test_elementwise_fusion_output_shape():
    fusion = QueryDocumentFusion(16, 16, 32, interaction_type="elementwise")
    query = torch.randn(4, 16)
    document = torch.randn(4, 16)

    out = fusion(query, document)

    assert out["fused"].shape == (4, 32)
    assert out["query_projected"].shape == (4, 32)
    assert out["document_projected"].shape == (4, 32)


def test_elementwise_features_arithmetic():
    query = torch.randn(3, 8)
    document = torch.randn(3, 8)

    features = elementwise_features(query, document)

    assert torch.allclose(features["product"], query * document)
    assert torch.allclose(features["absolute_difference"], (query - document).abs())


def test_identical_query_document_has_zero_difference_feature():
    vector = torch.randn(2, 6)
    features = elementwise_features(vector, vector)

    assert torch.allclose(features["absolute_difference"], torch.zeros_like(vector))
    assert torch.allclose(features["product"], vector * vector)


def test_fusion_handles_different_input_widths():
    fusion = QueryDocumentFusion(8, 20, 16)
    out = fusion(torch.randn(2, 8), torch.randn(2, 20))
    assert out["fused"].shape == (2, 16)


@pytest.mark.parametrize(
    "mode", ["none", "concatenate", "elementwise", "dot", "token"]
)
def test_all_interaction_modes_produce_expected_width(mode):
    fusion = QueryDocumentFusion(16, 16, 32, interaction_type=mode)
    out = fusion(torch.randn(5, 16), torch.randn(5, 16))
    assert out["fused"].shape == (5, 32)


def test_fusion_rejects_mismatched_batch_sizes():
    fusion = QueryDocumentFusion(8, 8, 16)
    with pytest.raises(ValueError, match="batch mismatch"):
        fusion(torch.randn(3, 8), torch.randn(4, 8))


def test_fusion_rejects_non_2d_input():
    fusion = QueryDocumentFusion(8, 8, 16)
    with pytest.raises(ValueError, match="2-D"):
        fusion(torch.randn(3, 8, 2), torch.randn(3, 8))


def test_fusion_is_differentiable():
    fusion = QueryDocumentFusion(8, 8, 16)
    query = torch.randn(2, 8, requires_grad=True)
    document = torch.randn(2, 8, requires_grad=True)

    fusion(query, document)["fused"].sum().backward()

    assert query.grad is not None
    assert document.grad is not None


def test_from_config_uses_config_dimensions():
    fusion = QueryDocumentFusion.from_config(12, 14, FusionConfig(fusion_dim=28))
    out = fusion(torch.randn(2, 12), torch.randn(2, 14))
    assert out["fused"].shape == (2, 28)


def test_fusion_parameter_count_matches_tensor_sum():
    fusion = QueryDocumentFusion(8, 8, 16)
    expected = sum(p.numel() for p in fusion.parameters())
    assert expected == sum(
        p.numel() for p in fusion.parameters() if p.requires_grad
    )
    assert expected > 0


def test_single_document_fusion_still_works():
    fusion = QueryDocumentFusion(8, 8, 16)
    out = fusion(torch.randn(1, 8), torch.randn(1, 8))
    assert out["fused"].shape == (1, 16)
