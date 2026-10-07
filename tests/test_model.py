"""Tests for the reranker model, encoder sharing and parameter accounting."""

from __future__ import annotations

import pytest
import torch

from embeddinggemma_reranker.model import (
    DocumentEncoder,
    EmbeddingGemma2Reranker,
    QueryEncoder,
    build_reranker,
)
from embeddinggemma_reranker.statistics import (
    breakdown,
    build_report,
    count_parameters,
    count_trainable,
    format_report,
)

from .conftest import tiny_config


def test_reranker_scores_shape(model):
    out = model.score("a query", ["doc one", "doc two", "doc three"])
    assert out.scores.shape == (3,)
    assert out.representations.shape[0] == 3


def test_query_and_document_encoders_share_one_backbone(model):
    """A 270M tower must not be duplicated in memory by default."""
    assert model.shares_backbone
    assert model.query_encoder.backbone is model.document_encoder.backbone


def test_query_encoder_is_encoded_once_per_call(model):
    """The query is tiled across candidates, not re-encoded N times."""
    calls = []
    original = model.backbone.forward

    def counting_forward(*args, **kwargs):
        calls.append(kwargs.get("input_ids", args[0] if args else None))
        return original(*args, **kwargs)

    model.backbone.forward = counting_forward
    try:
        model.clear_cache()
        model.score("the same query", [f"doc {i}" for i in range(5)])
    finally:
        model.backbone.forward = original

    # One query pass + one batched document pass, not five query passes.
    assert len(calls) == 2


def test_backbone_hidden_states_are_used(model):
    out = model.score("query text", ["document text"], return_details=True)
    assert "fused" in out.details
    assert out.details["fused"].shape == out.representations.shape


def test_native_similarity_matches_backbone_embeddings(model):
    """The 768-d embedding path is still computed for baseline comparison."""
    out = model.score("query", ["doc"])
    assert out.native_similarity is not None
    assert out.native_similarity.shape == (1,)
    # Unit-norm embeddings give a cosine in [-1, 1].
    assert -1.01 <= out.native_similarity.item() <= 1.01


def test_reranker_rejects_empty_document_list(model):
    with pytest.raises(ValueError, match="empty candidate list"):
        model.score("query", [])


def test_reranker_handles_single_document(model):
    out = model.score("query", ["only document"])
    assert out.scores.shape == (1,)


def test_variable_document_counts_produce_consistent_shapes(model):
    for count in (1, 2, 7, 16):
        out = model.score("query", [f"doc {i}" for i in range(count)])
        assert out.scores.shape == (count,)
        assert torch.isfinite(out.scores).all()


def test_scores_are_unbounded_raw_logits(model):
    """A sigmoid must not be baked into the head's output."""
    out = model.score("query", [f"doc {i}" for i in range(8)])
    assert out.scores.shape == (8,)
    # Any single forced-0..1 head would make these exactly 0.0 or 1.0.
    assert not torch.all((out.scores == 0) | (out.scores == 1))


def test_deterministic_scoring_in_eval_mode(model):
    model.eval()
    with torch.no_grad():
        first = model.score("query", ["a", "b", "c"]).scores
        second = model.score("query", ["a", "b", "c"]).scores
    assert torch.allclose(first, second, atol=1e-6)


def test_token_interaction_path_runs(interaction_model):
    out = interaction_model.score("query text", ["doc one", "doc two"])
    assert out.scores.shape == (2,)
    assert torch.isfinite(out.scores).all()


def test_confidence_head_produces_value(model):
    out = model.score("query", ["a", "b", "c"])
    confidence = model.confidence(out.representations, out.scores)
    assert confidence.shape == (1,)
    assert torch.isfinite(confidence).all()


def test_confidence_fallback_without_head():
    config = tiny_config()
    config.heads.confidence_head = False
    config.validate()
    model = build_reranker(config)
    assert model.confidence_head is None

    out = model.score("query", ["a", "b"])
    confidence = model.confidence(out.representations, out.scores)
    assert confidence.shape == (1,)


def test_multilingual_text_round_trips(model):
    """No English-only preprocessing: other scripts must score without error."""
    for text in ("数据库连接", "¿Cómo configurar?", "Данные конфигурация", "設定方法"):
        out = model.score(text, [text, "unrelated"])
        assert out.scores.shape == (2,)
        assert torch.isfinite(out.scores).all()


def test_utf8_text_does_not_crash_backbone(model):
    for text in ("emoji 🚀🚀", "עברית", "日本語のテキスト"):
        encoded = model.document_encoder([text])
        assert torch.isfinite(encoded.pooled).all()


def test_titles_are_applied_to_documents(model):
    model.config.prompts.enabled = True
    formatted = model.document_encoder.format_inputs(
        ["body text"], ["A Title"]
    )
    assert formatted[0].startswith("title: A Title | text: ")
    model.config.prompts.enabled = False


def test_prompt_templates_disabled_returns_raw_text(model):
    out = model.query_encoder.format_inputs(["plain query"])
    assert out == ["plain query"]


def test_backbone_is_shared_not_duplicated(model):
    total = count_parameters(model)
    backbone = count_parameters(model.backbone)
    # One backbone instance, so it must be less than half of the total.
    assert backbone < total


def test_parameter_breakdown_is_consistent(model):
    stats = breakdown(model)

    components = (
        stats["fusion"]
        + stats["interaction"]
        + stats["reranker_layers"]
        + stats["relevance_head"]
        + stats["confidence_head"]
    )
    assert stats["added_parameters"] == components
    assert stats["total_parameters"] == stats["backbone"] + stats["added_parameters"]
    assert (
        stats["trainable_parameters"] + stats["frozen_parameters"]
        == stats["total_parameters"]
    )


def test_parameter_counts_are_real_not_hardcoded(model):
    stats = breakdown(model)
    manual = sum(p.numel() for p in model.parameters())
    assert stats["total_parameters"] == manual


def test_trainable_percentage_is_consistent(model):
    stats = breakdown(model)
    expected = 100.0 * stats["trainable_parameters"] / stats["total_parameters"]
    assert stats["trainable_percentage"] == pytest.approx(expected, abs=1e-3)


def test_report_rendering_includes_every_section(model):
    report = build_report(model, train_mode="reranker_only")
    text = format_report(report)
    for label in (
        "backbone",
        "Fusion",
        "Reranker layers",
        "Relevance head",
        "Confidence head",
        "Total added parameters",
        "Trainable parameters",
    ):
        assert label in text


def test_query_document_representations_differ_for_different_text(model):
    """Sanity: the backbone is not collapsing everything to one vector."""
    out = model.score("alpha query", ["alpha doc", "beta doc"])
    assert not torch.allclose(out.representations[0], out.representations[1])


def test_backbone_shapes_match_config(model):
    assert model.backbone_hidden == model.config.backbone.hidden_size
    assert model.native_embedding_dim == model.config.backbone.embedding_dim


def test_encoder_prefixes_are_correct(model):
    assert isinstance(model.query_encoder, QueryEncoder)
    assert isinstance(model.document_encoder, DocumentEncoder)
    assert QueryEncoder.prefix == "query"
    assert DocumentEncoder.prefix == "document"
