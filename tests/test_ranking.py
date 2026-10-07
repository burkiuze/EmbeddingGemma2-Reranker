"""Tests for ranking, chunking, batching and abstention policy."""

from __future__ import annotations

import pytest
import torch

from embeddinggemma_reranker.config import PolicyConfig
from embeddinggemma_reranker.inference import (
    EmbeddingGemma2RerankerInference,
    aggregate_chunk_scores,
    chunk_document,
)
from embeddinggemma_reranker.policy import (
    RankingPolicy,
    build_ranking,
    sort_scores,
)

from .conftest import tiny_config


# --------------------------------------------------------------- ranking
def test_sort_scores_orders_descending():
    order = sort_scores(torch.tensor([0.5, -1.0, 2.0]))
    assert order.tolist() == [2, 0, 1]


def test_sort_scores_is_stable_for_ties():
    order = sort_scores(torch.tensor([1.0, 1.0, 1.0]))
    assert order.tolist() == [0, 1, 2]


def test_build_ranking_preserves_original_indices():
    scores = torch.tensor([0.1, 5.0, 2.0])
    result = build_ranking(scores)
    assert result["order"] == [1, 2, 0]
    assert result["ranking"][0]["index"] == 1
    assert result["ranking"][0]["rank"] == 1


def test_build_ranking_preserves_custom_ids():
    scores = torch.tensor([0.1, 5.0])
    result = build_ranking(scores, ["doc-a", "doc-b"])
    assert result["order"] == ["doc-b", "doc-a"]


def test_build_ranking_rejects_length_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        build_ranking(torch.tensor([1.0, 2.0]), ["only-one"])


def test_build_ranking_rejects_empty():
    with pytest.raises(ValueError, match="zero scores"):
        build_ranking(torch.zeros(0))


def test_build_ranking_returns_one_based_ranks():
    result = build_ranking(torch.tensor([3.0, 1.0, 2.0]))
    assert [item["rank"] for item in result["ranking"]] == [1, 2, 3]


# --------------------------------------------------------------- policy
def test_policy_accepts_by_default():
    policy = RankingPolicy(PolicyConfig())
    decision = policy.evaluate(top_score=torch.tensor(-5.0))
    assert decision.accepted
    assert policy.is_noop


def test_policy_rejects_low_relevance():
    policy = RankingPolicy(PolicyConfig(minimum_relevance=0.5))
    decision = policy.evaluate(top_score=torch.tensor(0.1))
    assert not decision.accepted
    assert decision.violated == "minimum_relevance"


def test_policy_rejects_small_margin():
    policy = RankingPolicy(PolicyConfig(minimum_margin=1.0))
    decision = policy.evaluate(
        top_score=torch.tensor(5.0), margin=torch.tensor(0.2)
    )
    assert not decision.accepted
    assert decision.violated == "minimum_margin"


def test_policy_rejects_low_confidence():
    policy = RankingPolicy(PolicyConfig(minimum_confidence=0.7))
    decision = policy.evaluate(confidence=torch.tensor(0.3))
    assert not decision.accepted
    assert decision.violated == "minimum_confidence"


def test_policy_treats_missing_signal_as_unmet():
    """A configured threshold with no signal must not silently accept."""
    policy = RankingPolicy(PolicyConfig(minimum_confidence=0.5))
    decision = policy.evaluate(top_score=torch.tensor(10.0))
    assert not decision.accepted
    assert "could not be evaluated" in decision.reason


# ---------------------------------------------------------------- chunking
@pytest.mark.parametrize(
    "aggregation,expected",
    [
        ("max", 3.0),
        ("mean", 2.0),
        ("top_k_mean", 2.5),
    ],
)
def test_aggregate_chunk_scores(aggregation, expected):
    scores = [1.0, 3.0, 2.0, 0.0]
    if aggregation == "top_k_mean":
        assert aggregate_chunk_scores(scores, aggregation, top_k=2) == pytest.approx(expected)
    else:
        assert aggregate_chunk_scores(scores, aggregation) == pytest.approx(expected)


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError, match="empty chunk score list"):
        aggregate_chunk_scores([])


def test_aggregate_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown chunk aggregation"):
        aggregate_chunk_scores([1.0], "median")


def test_chunk_document_returns_single_chunk_for_short_text(model):
    chunks = chunk_document("short text", model.backbone.tokenizer, max_tokens=64)
    assert chunks == ["short text"]


def test_chunk_document_splits_long_text(model):
    long_text = "word " * 400
    chunks = chunk_document(long_text, model.backbone.tokenizer, max_tokens=32, max_chunks=5)
    assert len(chunks) > 1
    assert len(chunks) <= 5


def test_chunk_document_respects_max_chunks(model):
    chunks = chunk_document(
        "word " * 1000, model.backbone.tokenizer, max_tokens=16, max_chunks=3
    )
    assert len(chunks) == 3


def test_chunk_document_validates_arguments():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_document("text", None, max_tokens=0)


# -------------------------------------------------------------- inference
def test_rerank_scores_every_document(model):
    inference = EmbeddingGemma2RerankerInference(model)
    documents = ["first", "second", "third", "fourth"]
    response = inference.rerank("query", documents)

    assert len(response.results) == 4
    assert sorted(response.ranking) == [0, 1, 2, 3]


def test_rerank_returns_descending_scores(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank("query", ["a", "b", "c", "d"])
    scores = [item["score"] for item in response.results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_ranks_are_contiguous_from_one(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank("query", ["a", "b", "c"])
    assert [item["rank"] for item in response.results] == [1, 2, 3]


def test_rerank_rejects_empty_documents(model):
    inference = EmbeddingGemma2RerankerInference(model)
    with pytest.raises(ValueError, match="empty candidate list"):
        inference.rerank("query", [])


def test_rerank_rejects_non_string_query(model):
    inference = EmbeddingGemma2RerankerInference(model)
    with pytest.raises(TypeError, match="must be a string"):
        inference.rerank(123, ["doc"])


def test_rerank_preserves_document_ids(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank(
        "query", ["a", "b", "c"], document_ids=["alpha", "beta", "gamma"]
    )
    assert sorted(response.ranking) == ["alpha", "beta", "gamma"]


def test_rerank_top_k_limits_results(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank("query", ["a", "b", "c", "d", "e"], top_k=2)
    assert len(response.results) == 2


def test_rerank_reports_metrics(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank("query", ["a", "b", "c"])
    assert response.metrics["num_documents"] == 3
    assert response.metrics["latency_seconds"] >= 0


def test_rerank_abstains_when_policy_not_met(model):
    config = tiny_config()
    config.policy = PolicyConfig(minimum_relevance=1000.0)
    config.validate()

    from embeddinggemma_reranker.model import build_reranker

    strict_model = build_reranker(config)
    inference = EmbeddingGemma2RerankerInference(strict_model, config)
    response = inference.rerank("query", ["a", "b", "c"])

    assert response.insufficient_confidence
    assert response.abstention_reason


def test_rerank_does_not_abstain_by_default(model):
    inference = EmbeddingGemma2RerankerInference(model)
    response = inference.rerank("query", ["a", "b", "c"])
    assert not response.insufficient_confidence


def test_rerank_chunking_is_reported(model):
    config = tiny_config()
    config.chunking.enabled = True
    config.chunking.max_tokens = 16
    config.chunking.max_chunks = 3
    config.training.max_document_length = 16
    config.validate()

    from embeddinggemma_reranker.model import build_reranker

    chunked_model = build_reranker(config)
    inference = EmbeddingGemma2RerankerInference(chunked_model, config)
    response = inference.rerank("query", ["a" * 400, "short"])

    # Truncation must never be silent.
    assert response.truncated_documents
    assert any(item.chunked for item in response.results)


def test_rerank_batch(model):
    inference = EmbeddingGemma2RerankerInference(model)
    responses = inference.rerank_batch(
        ["q1", "q2"], [["a", "b"], ["c", "d", "e"]]
    )
    assert len(responses) == 2
    assert len(responses[0].results) == 2
    assert len(responses[1].results) == 3


def test_rerank_batch_rejects_length_mismatch(model):
    inference = EmbeddingGemma2RerankerInference(model)
    with pytest.raises(ValueError, match="must have the same length"):
        inference.rerank_batch(["q1", "q2"], [["a"]])


def test_response_serializes_to_plain_dict(model):
    inference = EmbeddingGemma2RerankerInference(model)
    payload = inference.rerank("query", ["a", "b"]).to_dict()
    assert set(payload) >= {"query", "ranking", "results", "insufficient_confidence"}


# ---------------------------------------------------------------- batching
def test_batched_collator_pads_variable_candidate_counts(model, examples):
    from training.collator import RerankerCollator

    collator = RerankerCollator(
        tokenizer=model.backbone.tokenizer,
        max_query_length=24,
        max_document_length=32,
        prompts=model.config.prompts,
    )
    batch = collator(examples)

    assert batch.batch_size == 3
    assert batch.num_candidates == 3  # longest list wins
    assert batch.document_mask.sum().item() == 2 + 3 + 3


def test_document_mask_excludes_padding(model, examples):
    from training.collator import RerankerCollator

    collator = RerankerCollator(
        tokenizer=model.backbone.tokenizer,
        max_query_length=24,
        max_document_length=32,
    )
    batch = collator(examples)

    # Row 1 has only two real candidates; the third slot must be masked off.
    assert bool(batch.document_mask[1, 0])
    assert bool(batch.document_mask[1, 1])
    assert not bool(batch.document_mask[1, 2])


def test_flatten_documents_produces_flat_batch(model, examples):
    from training.collator import RerankerCollator

    collator = RerankerCollator(
        tokenizer=model.backbone.tokenizer,
        max_query_length=24,
        max_document_length=32,
    )
    batch = collator(examples)
    flat = batch.flatten_documents()

    assert flat["input_ids"].shape[0] == batch.batch_size * batch.num_candidates
    assert flat["input_ids"].dim() == 2


def test_collator_rejects_empty_batch(model):
    from training.collator import RerankerCollator

    collator = RerankerCollator(tokenizer=model.backbone.tokenizer)
    with pytest.raises(ValueError, match="empty batch"):
        collator([])
