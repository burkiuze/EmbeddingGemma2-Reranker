"""Tests for the configurable ranking losses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embeddinggemma_reranker.config import ConfigError
from training.losses import (
    bce_loss,
    compute_loss,
    distillation_kl_loss,
    listwise_loss,
    mse_loss,
    pairwise_margin_loss,
)


def make_batch(relevance, teacher=None):
    relevance_tensor = torch.tensor(relevance, dtype=torch.float32)
    batch = SimpleNamespace(
        relevance=relevance_tensor,
        binary_labels=(relevance_tensor > 0).to(torch.float32),
        pair_mask=torch.ones_like(relevance_tensor, dtype=torch.bool),
        teacher_scores=None
        if teacher is None
        else torch.tensor(teacher, dtype=torch.float32),
    )
    return batch


def test_pairwise_margin_prefers_correct_order():
    relevance = torch.tensor([[1.0, 0.0]])
    good = torch.tensor([[3.0, -1.0]])
    bad = torch.tensor([[-1.0, 3.0]])

    good_loss = pairwise_margin_loss(good, relevance, margin=0.1).loss
    bad_loss = pairwise_margin_loss(bad, relevance, margin=0.1).loss

    assert good_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert bad_loss.item() > 0.1


def test_pairwise_margin_returns_zero_metric_when_no_positives():
    """No positive means no pairs; the loss must be 0, not NaN."""
    scores = torch.tensor([[1.0, 2.0]])
    relevance = torch.tensor([[0.0, 0.0]])
    out = pairwise_margin_loss(scores, relevance)
    assert out.loss.item() == pytest.approx(0.0)
    assert out.metrics["num_pairs"] == 0


def test_pairwise_margin_is_differentiable():
    scores = torch.randn(2, 4, requires_grad=True)
    relevance = torch.tensor([[2.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    out = pairwise_margin_loss(scores, relevance)
    out.loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_bce_loss_decreases_with_correct_predictions():
    labels = torch.tensor([[1.0, 0.0]])
    confident_correct = torch.tensor([[6.0, -6.0]])
    confident_wrong = torch.tensor([[-6.0, 6.0]])

    good = bce_loss(confident_correct, labels).loss
    bad = bce_loss(confident_wrong, labels).loss

    assert good.item() < bad.item()


def test_bce_loss_ignores_masked_entries():
    labels = torch.tensor([[1.0, 0.0]])
    scores = torch.tensor([[10.0, -10.0]])
    mask = torch.tensor([[True, False]])

    out = bce_loss(scores, labels, mask=mask)
    assert out.metrics["num_supervised"] == 1


def test_listwise_loss_prefers_correct_distribution():
    relevance = torch.tensor([[0.0, 0.0, 2.0]])
    good = torch.tensor([[-2.0, -2.0, 5.0]])
    bad = torch.tensor([[5.0, 4.0, -2.0]])

    good_loss = listwise_loss(good, relevance).loss
    bad_loss = listwise_loss(bad, relevance).loss

    assert good_loss.item() < bad_loss.item()


def test_listwise_loss_excludes_padding_from_softmax():
    """A padded candidate with a huge score must not influence the loss."""
    relevance = torch.tensor([[0.0, 2.0]])
    mask = torch.tensor([[False, True]])

    with_pad = torch.tensor([[100.0, 1.0]])
    without_pad = torch.tensor([[-100.0, 1.0]])

    loss_masked = listwise_loss(with_pad, relevance, mask=mask).loss
    loss_clean = listwise_loss(without_pad, relevance, mask=mask).loss

    assert torch.isfinite(loss_masked)
    assert loss_masked.item() == pytest.approx(loss_clean.item(), abs=1e-5)


def test_mse_loss_measures_squared_error():
    scores = torch.tensor([[2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    out = mse_loss(scores, targets)
    assert out.loss.item() == pytest.approx(0.5)


def test_distillation_kl_is_zero_when_teacher_matches_student():
    teacher = torch.tensor([[2.0, 0.0]])
    student = torch.tensor([[2.0, 0.0]])
    out = distillation_kl_loss(student, teacher)
    assert out.loss.item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_kl_positive_for_mismatched_distributions():
    out = distillation_kl_loss(
        torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 2.0]])
    )
    assert out.loss.item() > 0.0


def test_compute_loss_rejects_unknown_name():
    batch = make_batch([[1.0, 0.0]])
    with pytest.raises(ConfigError, match="unknown loss"):
        compute_loss("contrastive", torch.randn(1, 2), batch)


@pytest.mark.parametrize(
    "name", ["bce", "pairwise_margin", "listwise", "mse"]
)
def test_compute_loss_dispatch(name):
    batch = make_batch([[2.0, 0.0]])
    scores = torch.randn(1, 2, requires_grad=True)
    out = compute_loss(name, scores, batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert scores.grad is not None


def test_compute_loss_adds_distillation_when_teacher_present():
    batch = make_batch([[2.0, 0.0]], teacher=[[5.0, 1.0]])
    scores = torch.randn(1, 2, requires_grad=True)
    out = compute_loss("listwise", scores, batch, distillation_weight=0.5)
    assert "distill_kl_loss" in out.metrics
    assert out.metrics["distill_weight"] == 0.5


def test_distill_kl_refuses_to_invent_teacher_scores():
    """No teacher data means an error, never a fabricated target."""
    batch = make_batch([[1.0, 0.0]])
    with pytest.raises(ConfigError, match="fabricate teacher labels"):
        compute_loss("distill_kl", torch.randn(1, 2), batch)


def test_losses_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        bce_loss(torch.randn(2, 3), torch.randn(2, 2))


def test_loss_gradients_flow_through_scores_only():
    scores = torch.randn(3, 4, requires_grad=True)
    relevance = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    )
    pairwise_margin_loss(scores, relevance).loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0
