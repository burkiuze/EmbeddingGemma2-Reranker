"""Ranking metrics.

Standard IR metrics, implemented directly so evaluation has no dependency on
``evaluate``/``mteb`` and so each one is unit-testable against known values.

A recurring trap in reranker evaluation is slicing the ranking to ``k`` while
keeping the full relevance vector. Every metric here takes the full ranking and
only *reads* the first ``k`` entries, so ``NDCG@10`` over a 20-document list is
computed correctly rather than crashing or silently biasing the result.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _validate(ranking: Sequence[int], relevance: Sequence[int]) -> None:
    if len(ranking) != len(relevance):
        raise ValueError(
            f"ranking and relevance must have equal length, got "
            f"{len(ranking)} and {len(relevance)}"
        )
    if not ranking:
        raise ValueError("cannot score an empty ranking")


def hits_at_k(ranking: Sequence[int], relevance: Sequence[int], k: int) -> float:
    """1.0 if any of the top ``k`` results is relevant, else 0.0."""
    _validate(ranking, relevance)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = {i for i, grade in enumerate(relevance) if grade > 0}
    if not relevant:
        return 0.0
    for position, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1.0
    return 0.0


def precision_at_k(ranking: Sequence[int], relevance: Sequence[int], k: int) -> float:
    """Relevant documents in the top ``k`` divided by ``k``.

    Uses ``min(k, len(ranking))`` as the denominator so a list shorter than ``k``
    is not silently penalised for not existing.
    """
    _validate(ranking, relevance)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    cutoff = min(k, len(ranking))
    relevant = {i for i, grade in enumerate(relevance) if grade > 0}
    hits = sum(1 for doc in ranking[:cutoff] if doc in relevant)
    return hits / cutoff


def recall_at_k(ranking: Sequence[int], relevance: Sequence[int], k: int) -> float:
    """Fraction of all relevant documents found in the top ``k``."""
    _validate(ranking, relevance)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = {i for i, grade in enumerate(relevance) if grade > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc in ranking[:k] if doc in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranking: Sequence[int], relevance: Sequence[int]) -> float:
    """Reciprocal rank of the first relevant document."""
    _validate(ranking, relevance)
    relevant = {i for i, grade in enumerate(relevance) if grade > 0}
    for position, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def average_precision(
    ranking: Sequence[int], relevance: Sequence[int]
) -> float:
    """Mean of precision at each relevant hit position."""
    _validate(ranking, relevance)
    relevant = {i for i, grade in enumerate(relevance) if grade > 0}
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0
    for position, doc in enumerate(ranking, start=1):
        if doc in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def dcg_at_k(
    ranking: Sequence[int],
    relevance: Sequence[int],
    k: int,
    graded: bool = True,
) -> float:
    """Discounted cumulative gain over the top ``k``."""
    _validate(ranking, relevance)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    gains = []
    for position, doc in enumerate(ranking[:k], start=1):
        grade = relevance[doc]
        gains.append((2**grade - 1) if graded else float(grade > 0))
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(
    ranking: Sequence[int], relevance: Sequence[int], k: int, graded: bool = True
) -> float:
    """Normalized DCG, binary relevance by default."""
    _validate(ranking, relevance)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ideal = sorted(range(len(relevance)), key=lambda i: relevance[i], reverse=True)
    ideal_dcg = dcg_at_k(ideal, relevance, k, graded=graded)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranking, relevance, k, graded=graded) / ideal_dcg


def top1_accuracy(ranking: Sequence[int], relevance: Sequence[int]) -> float:
    """1.0 when the best-ranked document is relevant."""
    _validate(ranking, relevance)
    return 1.0 if relevance[ranking[0]] > 0 else 0.0


def pairwise_accuracy(
    ranking: Sequence[int], relevance: Sequence[int]
) -> float:
    """Fraction of (relevant, irrelevant) pairs ordered correctly.

    Ties in relevance are skipped, so graded data with equal grades does not
    inflate or deflate the score.
    """
    _validate(ranking, relevance)
    relevant = [doc for doc in ranking if relevance[doc] > 0]
    irrelevant = [doc for doc in ranking if relevance[doc] <= 0]
    if not relevant or not irrelevant:
        return 0.0

    total = 0
    correct = 0
    for rel in relevant:
        for irr in irrelevant:
            total += 1
            if ranking.index(rel) < ranking.index(irr):
                correct += 1
    return correct / total


def evaluate_rankings(
    rankings: Sequence[Sequence[int]],
    relevances: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 5, 10),
    graded: bool = True,
) -> Dict[str, float]:
    """Aggregate metrics over many queries.

    Parameters
    ----------
    rankings: one permutation per query, containing every document index.
    relevances: graded relevance aligned with the document indices.
    ks: cutoffs for the ``@k`` metrics. The ranking is **not** truncated before
        computing them, which avoids the length-mismatch class of bug.
    """
    if len(rankings) != len(relevances):
        raise ValueError(
            f"rankings ({len(rankings)}) and relevances ({len(relevances)}) "
            "must have the same number of queries"
        )
    if not rankings:
        raise ValueError("cannot evaluate an empty set of rankings")

    for query_index, (ranking, relevance) in enumerate(zip(rankings, relevances)):
        _validate(ranking, relevance)
        if sorted(ranking) != list(range(len(relevance))):
            raise ValueError(
                f"query {query_index}: ranking must be a permutation of "
                f"0..{len(relevance) - 1}"
            )

    results: Dict[str, float] = {}

    def mean(name: str, values: List[float]) -> None:
        results[name] = float(sum(values) / len(values)) if values else 0.0

    mean("mrr", [reciprocal_rank(r, g) for r, g in zip(rankings, relevances)])
    mean("map", [average_precision(r, g) for r, g in zip(rankings, relevances)])
    mean("top1_accuracy", [top1_accuracy(r, g) for r, g in zip(rankings, relevances)])
    mean(
        "pairwise_accuracy",
        [pairwise_accuracy(r, g) for r, g in zip(rankings, relevances)],
    )

    for k in ks:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        mean(f"mrr@{k}", [reciprocal_rank(r, g) for r, g in zip(rankings, relevances)])
        mean(
            f"precision@{k}",
            [precision_at_k(r, g, k) for r, g in zip(rankings, relevances)],
        )
        mean(f"recall@{k}", [recall_at_k(r, g, k) for r, g in zip(rankings, relevances)])
        mean(
            f"ndcg@{k}",
            [ndcg_at_k(r, g, k, graded=graded) for r, g in zip(rankings, relevances)],
        )
        mean(
            f"hits@{k}",
            [hits_at_k(r, g, k) for r, g in zip(rankings, relevances)],
        )

    results["num_queries"] = float(len(rankings))
    results["mean_candidates"] = float(
        sum(len(r) for r in rankings) / len(rankings)
    )
    return results


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for a per-query metric.

    Use this before claiming a reranker beats a baseline: single-run differences
    on small query sets are frequently inside their own noise.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty value list")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    means = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, array.size, array.size)
        means[index] = array[sample].mean()

    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "lower": float(np.percentile(means, 100 * alpha)),
        "upper": float(np.percentile(means, 100 * (1 - alpha))),
        "confidence": confidence,
        "num_samples": int(array.size),
    }
