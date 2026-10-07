"""Bi-encoder baseline: EmbeddingGemma query embedding vs cosine similarity.

This exists to answer one question honestly: *does the learned reranker actually
beat what the source model already does?* Until a trained reranker is evaluated
against this, no improvement claim is justified.

The baseline deliberately uses the backbone's **native 768-d embedding** with the
upstream mean pooling and L2 normalization, so it reproduces the published
EmbeddingGemma retrieval behaviour exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from embeddinggemma_reranker.model import EmbeddingGemma2Reranker

from .metrics import evaluate_rankings


@dataclass
class BaselineResult:
    """Ranking produced by the cosine baseline."""

    rankings: List[List[int]] = field(default_factory=list)
    scores: List[List[float]] = field(default_factory=list)
    query_embeddings: Optional[torch.Tensor] = None
    document_embeddings: Optional[torch.Tensor] = None
    wall_seconds: float = 0.0

    def as_metrics(self, relevances: Sequence[Sequence[int]], ks=(1, 5, 10)):
        return evaluate_rankings(self.rankings, relevances, ks=ks)


class BiEncoderBaseline:
    """Cosine-similarity retrieval over the backbone's native embeddings."""

    def __init__(self, model: EmbeddingGemma2Reranker) -> None:
        self.model = model

    @torch.no_grad()
    def encode_queries(self, queries: Sequence[str]) -> torch.Tensor:
        if not queries:
            raise ValueError("cannot encode an empty query list")
        return self.model.query_encoder(list(queries)).embedding

    @torch.no_grad()
    def encode_documents(self, documents: Sequence[str]) -> torch.Tensor:
        if not documents:
            raise ValueError("cannot encode an empty document list")
        return self.model.document_encoder(list(documents)).embedding

    @staticmethod
    def score(
        query_embeddings: torch.Tensor, document_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity matrix ``(Q, D)``."""
        if query_embeddings.dim() != 2 or document_embeddings.dim() != 2:
            raise ValueError("embeddings must be 2-D (N, D)")
        if query_embeddings.shape[1] != document_embeddings.shape[1]:
            raise ValueError(
                f"embedding widths differ: {query_embeddings.shape[1]} vs "
                f"{document_embeddings.shape[1]}"
            )
        query_norm = F.normalize(query_embeddings, p=2.0, dim=-1)
        document_norm = F.normalize(document_embeddings, p=2.0, dim=-1)
        return query_norm @ document_norm.T

    @torch.no_grad()
    def rank(
        self, query: str, documents: Sequence[str]
    ) -> List[int]:
        """Return document indices sorted by descending cosine similarity."""
        if not documents:
            raise ValueError("cannot rank an empty candidate list")

        query_embedding = self.encode_queries([query])
        document_embeddings = self.encode_documents(documents)
        scores = self.score(query_embedding, document_embeddings)[0]

        # Stable sort keeps ties in input order, matching the reranker's ordering.
        order = torch.argsort(scores, descending=True, stable=True)
        return order.tolist()

    @torch.no_grad()
    def rank_many(
        self, queries: Sequence[str], document_lists: Sequence[Sequence[str]]
    ) -> BaselineResult:
        """Rank many queries, each against its own candidate list."""
        if len(queries) != len(document_lists):
            raise ValueError(
                f"queries ({len(queries)}) and document lists "
                f"({len(document_lists)}) must have the same length"
            )

        started = time.perf_counter()
        rankings: List[List[int]] = []
        scores: List[List[float]] = []
        for query, documents in zip(queries, document_lists):
            if not documents:
                raise ValueError("empty candidate list passed to baseline")
            query_embedding = self.encode_queries([query])
            document_embeddings = self.encode_documents(documents)
            row = self.score(query_embedding, document_embeddings)[0]
            order = torch.argsort(row, descending=True, stable=True)
            rankings.append(order.tolist())
            scores.append(row.tolist())

        return BaselineResult(
            rankings=rankings,
            scores=scores,
            wall_seconds=time.perf_counter() - started,
        )


def compare_against_reranker(
    model: EmbeddingGemma2Reranker,
    queries: Sequence[str],
    document_lists: Sequence[Sequence[str]],
    relevances: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """Run both systems on identical inputs and report the deltas.

    A negative delta means the reranker lost; this function reports whatever it
    finds and does not filter for favourable numbers.
    """
    if not (len(queries) == len(document_lists) == len(relevances)):
        raise ValueError(
            "queries, document_lists and relevances must be the same length "
            f"(got {len(queries)}, {len(document_lists)}, {len(relevances)})"
        )

    baseline = BiEncoderBaseline(model)
    baseline_output = baseline.rank_many(queries, document_lists)
    baseline_metrics = evaluate_rankings(
        baseline_output.rankings, relevances, ks=ks
    )

    reranker_rankings: List[List[int]] = []
    started = time.perf_counter()
    for query, documents in zip(queries, document_lists):
        output = model.score(query, documents)
        order = torch.argsort(output.scores, descending=True, stable=True)
        reranker_rankings.append(order.tolist())
    reranker_seconds = time.perf_counter() - started

    reranker_metrics = evaluate_rankings(reranker_rankings, relevances, ks=ks)

    deltas: Dict[str, float] = {}
    for key, value in reranker_metrics.items():
        if isinstance(value, (int, float)) and key in baseline_metrics:
            deltas[key] = round(value - baseline_metrics[key], 6)

    return {
        "baseline": baseline_metrics,
        "reranker": reranker_metrics,
        "deltas": deltas,
        "baseline_wall_seconds": baseline_output.wall_seconds,
        "reranker_wall_seconds": reranker_seconds,
        "num_queries": len(queries),
    }


def format_comparison(report: Dict[str, Any], keys: Sequence[str]) -> str:
    """Render a comparison table. Signs are preserved; negatives stay negative."""
    baseline = report["baseline"]
    reranker = report["reranker"]
    deltas = report["deltas"]

    header = f"{'metric':<20}{'baseline':>12}{'reranker':>12}{'delta':>12}"
    lines = [header, "-" * len(header)]
    for key in keys:
        if key not in baseline:
            continue
        lines.append(
            f"{key:<20}{baseline[key]:>12.4f}{reranker.get(key, 0.0):>12.4f}"
            f"{deltas.get(key, 0.0):>+12.4f}"
        )
    lines.append("")
    lines.append(f"baseline wall  : {report['baseline_wall_seconds']:.3f}s")
    lines.append(f"reranker wall  : {report['reranker_wall_seconds']:.3f}s")
    lines.append(f"queries        : {report['num_queries']}")
    return "\n".join(lines)
