"""Tests for the relevance and confidence heads."""

from __future__ import annotations

import pytest
import torch

from embeddinggemma_reranker.config import HeadConfig
from embeddinggemma_reranker.heads import (
    ConfidenceHead,
    RelevanceHead,
    calibrate,
    confidence_from_scores,
)


def test_relevance_head_emits_one_raw_logit_per_pair():
    head = RelevanceHead(input_dim=16, hidden_dim=8)
    out = head(torch.randn(5, 16))
    assert out.shape == (5,)


def test_relevance_head_output_is_unbounded():
    """Training must not squash scores into [0, 1]."""
    head = RelevanceHead(input_dim=4, hidden_dim=4)
    large = torch.full((3, 4), 50.0)
    out = head(large)
    assert out.shape == (3,)
    # A sigmoid-free head can exceed 1.0; assert it is not clipped.
    assert out.abs().max() > 1.0


def test_relevance_head_accepts_extra_features():
    head = RelevanceHead(input_dim=8, hidden_dim=4, extra_features=3)
    out = head(torch.randn(2, 8), torch.randn(2, 3))
    assert out.shape == (2,)


def test_relevance_head_rejects_wrong_extra_width():
    head = RelevanceHead(input_dim=8, hidden_dim=4, extra_features=3)
    with pytest.raises(ValueError, match="extra feature width"):
        head(torch.randn(2, 8), torch.randn(2, 5))


def test_relevance_head_rejects_missing_extras():
    head = RelevanceHead(input_dim=8, hidden_dim=4, extra_features=3)
    with pytest.raises(ValueError, match="got none"):
        head(torch.randn(2, 8))


def test_confidence_score_statistics_shape():
    scores = torch.randn(4, 6)
    stats = ConfidenceHead.score_statistics(scores)
    assert stats.shape == (4, ConfidenceHead.num_statistics)


def test_confidence_score_statistics_single_candidate_has_zero_margin():
    """A one-candidate list has no top2; margin must be 0, not NaN."""
    stats = ConfidenceHead.score_statistics(torch.tensor([[2.5]]))
    assert stats.shape == (1, ConfidenceHead.num_statistics)
    assert stats[0, 1].item() == 0.0
    assert torch.isfinite(stats).all()


def test_confidence_score_statistics_values():
    scores = torch.tensor([[3.0, 1.0, 0.0]])
    stats = ConfidenceHead.score_statistics(scores)
    assert stats[0, 0].item() == pytest.approx(3.0)   # top
    assert stats[0, 1].item() == pytest.approx(2.0)   # margin
    assert stats[0, 3].item() == pytest.approx(4 / 3)  # mean


def test_confidence_score_statistics_rejects_empty():
    with pytest.raises(ValueError, match="empty candidate set"):
        ConfidenceHead.score_statistics(torch.zeros(1, 0))


def test_confidence_head_output_shape():
    head = ConfidenceHead(input_dim=16, hidden_dim=8)
    representations = torch.randn(2, 5, 16)
    scores = torch.randn(2, 5)
    assert head(representations, scores).shape == (2,)


def test_confidence_head_handles_single_candidate_list():
    head = ConfidenceHead(input_dim=8, hidden_dim=8)
    out = head(torch.randn(1, 1, 8), torch.randn(1, 1))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_confidence_head_rejects_shape_mismatch():
    head = ConfidenceHead(input_dim=8, hidden_dim=8)
    with pytest.raises(ValueError, match="shape mismatch"):
        head(torch.randn(2, 5, 8), torch.randn(2, 4))


def test_heads_are_trainable_modules():
    relevance = RelevanceHead(input_dim=4, hidden_dim=4)
    confidence = ConfidenceHead(input_dim=4, hidden_dim=4)
    total = sum(p.numel() for p in relevance.parameters())
    total += sum(p.numel() for p in confidence.parameters())
    assert total > 0
    assert all(p.requires_grad for p in relevance.parameters())


def test_from_config_dimensions():
    relevance = RelevanceHead.from_config(12, HeadConfig(relevance_hidden_dim=20))
    assert relevance(torch.randn(2, 12)).shape == (2,)

    confidence = ConfidenceHead.from_config(12, HeadConfig(confidence_hidden_dim=18))
    assert confidence(torch.randn(2, 4, 12), torch.randn(2, 4)).shape == (2,)


@pytest.mark.parametrize("method", ["sigmoid", "none"])
def test_calibrate_returns_expected_range(method):
    logits = torch.tensor([-3.0, 0.0, 3.0])
    out = calibrate(logits, method)
    if method == "sigmoid":
        assert out.min() >= 0.0 and out.max() <= 1.0
    else:
        assert torch.equal(out, logits)


def test_calibrate_rejects_unknown_method():
    with pytest.raises(ValueError, match="unsupported calibration"):
        calibrate(torch.zeros(1), "softmax")


def test_confidence_from_scores_peaked_distribution_is_higher():
    peaked = confidence_from_scores(torch.tensor([5.0, 0.0]), 2)
    flat = confidence_from_scores(torch.tensor([1.0, 0.9]), 2)
    assert peaked.item() > flat.item()


def test_confidence_from_scores_single_candidate_is_one():
    out = confidence_from_scores(torch.tensor([0.7]), 1)
    assert out.item() == pytest.approx(1.0)
