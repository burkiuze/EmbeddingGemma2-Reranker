"""Regression coverage for inference and training failures in the original repo."""

import pytest
import torch

from embeddinggemma_reranker.fusion import QueryDocumentFusion
from embeddinggemma_reranker.inference import EmbeddingGemma2RerankerInference
from embeddinggemma_reranker.interaction import TokenInteraction
from embeddinggemma_reranker.model import build_reranker
from evaluation.evaluate import evaluate_reranker, format_report, load_checkpoint
from training.losses import pairwise_margin_loss, listwise_loss, distillation_kl_loss
from training.trainer import RerankerTrainer
from .conftest import tiny_config


def test_dot_fusion_preserves_parameters_and_optimizer():
    fusion = QueryDocumentFusion(8, 8, 12, interaction_type="dot", dropout=0.0).eval()
    inputs = (torch.randn(2, 8), torch.randn(2, 8))
    before = tuple(id(p) for p in fusion.parameters())
    first = fusion(*inputs)["fused"]
    second = fusion(*inputs)["fused"]
    assert before == tuple(id(p) for p in fusion.parameters())
    assert torch.allclose(first, second)
    first.sum().backward()
    assert fusion.fusion_proj[0].weight.grad is not None


def test_projected_last_state_is_not_used_as_transformer_state(model):
    hidden = torch.randn(2, 3, 32)
    projected = torch.randn(2, 3, 48)
    assert torch.equal(model.backbone._select_layers((hidden, projected), 2, 3), hidden)


def test_native_embedding_ignores_padding(model):
    first = model.document_encoder(["short"]).embedding[0]
    batched = model.document_encoder(["short", "a much longer document"]).embedding[0]
    assert torch.allclose(first, batched, atol=1e-6)


def test_multi_layer_fusion_matches_selected_width():
    config = tiny_config()
    config.backbone.hidden_layer_ids = [0, 1]
    model = build_reranker(config)
    assert model.backbone_hidden == 64
    assert torch.isfinite(model.score("q", ["a", "b"]).scores).all()


def test_pairwise_loss_is_independent_of_candidate_order():
    scores = torch.tensor([[-1.0, 2.0, 0.0]])
    labels = torch.tensor([[2.0, 0.0, 1.0]])
    permutation = [2, 0, 1]
    original = pairwise_margin_loss(scores, labels).loss
    reordered = pairwise_margin_loss(scores[:, permutation], labels[:, permutation]).loss
    assert original.item() == pytest.approx(reordered.item())
    assert original.item() > 0


@pytest.mark.parametrize("loss", [listwise_loss, distillation_kl_loss])
def test_distribution_losses_remove_padding_from_targets(loss):
    scores = torch.tensor([[2.0, -1.0, 100.0]], requires_grad=True)
    labels = torch.tensor([[2.0, 0.0, 50.0]])
    mask = torch.tensor([[True, True, False]])
    padded = loss(scores, labels, mask=mask).loss
    unpadded = loss(scores[:, :2], labels[:, :2]).loss
    assert padded.item() == pytest.approx(unpadded.item(), abs=1e-6)
    padded.backward()
    assert scores.grad[0, 2].item() == 0.0


def test_interaction_ignores_masked_tokens():
    interaction = TokenInteraction(8, 8, 2, dropout=0).eval()
    q, d = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    qm = torch.tensor([[True, True, False]] * 2)
    dm = torch.tensor([[True, True, True, False]] * 2)
    first = interaction(q, d, qm, dm)["token_context"]
    q[:, -1] = 10000
    d[:, -1] = -10000
    second = interaction(q, d, qm, dm)["token_context"]
    assert torch.allclose(first, second, atol=1e-6)


def test_duplicate_document_ids_keep_correct_text(model):
    response = EmbeddingGemma2RerankerInference(model).rerank("q", ["first", "second"], document_ids=["same", "same"])
    assert sorted(item.document for item in response.results) == ["first", "second"]


def test_chunking_reports_only_dropped_content():
    config = tiny_config()
    config.chunking.enabled = True
    config.chunking.max_tokens = 8
    config.chunking.max_chunks = 3
    response = EmbeddingGemma2RerankerInference(build_reranker(config)).rerank("q", ["a" * 20, "b" * 30])
    assert all(item.chunked for item in response.results)
    assert response.truncated_documents == [1]


def test_training_ragged_batches_checkpoint_and_evaluation(tmp_path, examples):
    config = tiny_config()
    config.training.output_dir = str(tmp_path)
    config.training.epochs = 1
    config.training.gradient_accumulation_steps = 3
    model = build_reranker(config)
    backbone_before = {key: tensor.clone() for key, tensor in model.backbone.state_dict().items()}
    head_before = model.relevance_head.mlp[-1].weight.detach().clone()
    trainer = RerankerTrainer(model, config)
    summary = trainer.train(examples)
    assert summary.steps_completed == summary.steps == 1
    assert all(torch.equal(value, model.backbone.state_dict()[key]) for key, value in backbone_before.items())
    assert not torch.equal(head_before, model.relevance_head.mlp[-1].weight)
    restored = load_checkpoint(trainer.save(summary))
    with torch.no_grad():
        assert torch.allclose(model.score("q", ["a", "b"]).scores, restored.score("q", ["a", "b"]).scores)
    report = evaluate_reranker(restored, examples, measure_latency=True)
    assert "Baseline" in format_report(report)
    assert torch.isfinite(torch.tensor(report["confidence"]["ece"]))
