"""Evaluation entry points.

Runs the learned reranker and the bi-encoder baseline on identical inputs and
reports both plus the delta, together with latency/throughput when a timing flag is
passed. Confidence calibration is included whenever the model exposes a
confidence head.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from embeddinggemma_reranker.config import RerankerConfig
from embeddinggemma_reranker.model import EmbeddingGemma2Reranker

from .baseline import BiEncoderBaseline, compare_against_reranker, format_comparison
from .calibration import evaluate_confidence
from .metrics import bootstrap_ci, evaluate_rankings, reciprocal_rank

DEFAULT_KS = (1, 5, 10)


def grade_to_binary(relevance: Sequence[float], threshold: float = 0.0) -> List[int]:
    """Convert graded relevance to binary for metrics that need it."""
    return [1 if value > threshold else 0 for value in relevance]


def _query_inputs(
    examples: Sequence[Any],
) -> tuple[List[str], List[List[str]], List[List[int]]]:
    queries: List[str] = []
    document_lists: List[List[str]] = []
    relevances: List[List[int]] = []

    for example in examples:
        queries.append(example.query)
        document_lists.append(list(example.documents))
        relevances.append(list(example.relevance))
    return queries, document_lists, relevances


@torch.no_grad()
def evaluate_reranker(
    model: EmbeddingGemma2Reranker,
    examples: Sequence[Any],
    ks: Sequence[int] = DEFAULT_KS,
    measure_latency: bool = False,
    include_baseline: bool = True,
) -> Dict[str, Any]:
    """Score ``examples`` with the reranker and optionally with the baseline."""
    if not examples:
        raise ValueError("cannot evaluate an empty example list")

    model.eval()
    queries, document_lists, relevances = _query_inputs(examples)

    rankings: List[List[int]] = []
    per_query_rr: List[float] = []
    latencies: List[float] = []
    confidences: List[float] = []
    confidences_correct: List[bool] = []

    for query, documents, relevance in zip(queries, document_lists, relevances):
        started = time.perf_counter()
        output = model.score(query, documents)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        order = torch.argsort(output.scores, descending=True, stable=True).tolist()
        rankings.append(order)
        per_query_rr.append(reciprocal_rank(order, relevance))

        if model.confidence_head is not None:
            from .calibration import top_margin_from_scores

            confidence_logits = model.confidence(
                output.representations,
                output.scores,
            )
            confidence = float(torch.sigmoid(confidence_logits).item())
            confidences.append(confidence)
            confidences_correct.append(relevance[order[0]] > 0)

    report: Dict[str, Any] = {
        "reranker": evaluate_rankings(rankings, relevances, ks=ks),
    }

    if measure_latency:
        total_documents = sum(len(d) for d in document_lists)
        report["performance"] = {
            "num_queries": len(queries),
            "total_documents": total_documents,
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "median_latency_seconds": sorted(latencies)[len(latencies) // 2],
            "total_seconds": sum(latencies),
            "queries_per_second": len(queries) / sum(latencies)
            if sum(latencies) > 0
            else None,
            "documents_per_second": total_documents / sum(latencies)
            if sum(latencies) > 0
            else None,
        }

    if confidences:
        report["confidence"] = evaluate_confidence(confidences, confidences_correct)
        report["confidence"]["mrr_ci"] = bootstrap_ci(per_query_rr)

    if include_baseline:
        comparison = compare_against_reranker(
            model, queries, document_lists, relevances, ks=ks
        )
        report["baseline"] = comparison["baseline"]
        report["deltas"] = comparison["deltas"]
        report["baseline_wall_seconds"] = comparison["baseline_wall_seconds"]
        report["reranker_wall_seconds"] = comparison["reranker_wall_seconds"]

    return report


def format_report(report: Dict[str, Any], ks: Sequence[int] = DEFAULT_KS) -> str:
    """Human-readable summary."""
    lines: List[str] = []
    reranker = report.get("reranker", {})

    lines.append("Reranker metrics")
    header = f"  {'metric':<20}{'value':>12}"
    lines.append(header)
    lines.append("  " + "-" * len(header.strip()))
    for key in ("mrr", "map", "top1_accuracy", "pairwise_accuracy"):
        if key in reranker:
            lines.append(f"  {key:<20}{reranker[key]:>12.4f}")
    for k in ks:
        for key in (f"mrr@{k}", f"ndcg@{k}", f"recall@{k}", f"precision@{k}"):
            if key in reranker:
                lines.append(f"  {key:<20}{reranker[key]:>12.4f}")

    if "baseline" in report:
        lines.append("")
        lines.append("Baseline (EmbeddingGemma cosine) vs reranker")
        lines.append(format_comparison(
            {"baseline": report["baseline"], "reranker": reranker, "deltas": report["deltas"], "num_queries": reranker.get("num_queries", 0),
             "baseline_wall_seconds": report["baseline_wall_seconds"],
             "reranker_wall_seconds": report["reranker_wall_seconds"]},
            ["mrr", "map", "top1_accuracy"] + [f"ndcg@{k}" for k in ks],
        ))

    if "performance" in report:
        perf = report["performance"]
        lines.append("")
        lines.append("Performance")
        lines.append(f"  mean latency        : {perf['mean_latency_seconds'] * 1000:.2f} ms")
        lines.append(f"  documents/second    : {perf['documents_per_second']:.1f}")
        lines.append(f"  queries/second      : {perf['queries_per_second']:.1f}")

    if "confidence" in report:
        conf = report["confidence"]
        lines.append("")
        lines.append("Confidence calibration")
        lines.append(f"  ECE                 : {conf['ece']:.4f}")
        lines.append(f"  Brier               : {conf['brier']:.4f}")
        lines.append(f"  mean confidence     : {conf['mean_confidence']:.4f}")
        lines.append(f"  accuracy            : {conf['accuracy']:.4f}")

    return "\n".join(lines)


def evaluate_and_write(
    model: EmbeddingGemma2Reranker,
    examples: Sequence[Any],
    output_path: Optional[str | Path] = None,
    ks: Sequence[int] = DEFAULT_KS,
    measure_latency: bool = False,
) -> Dict[str, Any]:
    """Evaluate and optionally persist the report as JSON."""
    report = evaluate_reranker(
        model, examples, ks=ks, measure_latency=measure_latency
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_checkpoint(
    checkpoint_dir: str | Path,
    backbone: Optional[torch.nn.Module] = None,
) -> EmbeddingGemma2Reranker:
    """Rebuild a reranker from a training checkpoint directory."""
    from embeddinggemma_reranker.model import build_reranker

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "reranker_config.json"
    state_path = checkpoint_dir / "reranker_state.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"no reranker_config.json in {checkpoint_dir}")
    if not state_path.exists():
        raise FileNotFoundError(f"no reranker_state.pt in {checkpoint_dir}")

    config = RerankerConfig.from_json(config_path)
    model = build_reranker(config, backbone=backbone)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
