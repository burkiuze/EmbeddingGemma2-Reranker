#!/usr/bin/env python3
"""Basic reranking with the learned cross-encoder.

    python examples/rerank_documents.py

Uses the real checkpoint when available; falls back to the stub backbone with a
clear notice so the example always runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from embeddinggemma_reranker.config import RerankerConfig  # noqa: E402
from embeddinggemma_reranker.inference import EmbeddingGemma2RerankerInference  # noqa: E402
from embeddinggemma_reranker.model import build_reranker  # noqa: E402
from embeddinggemma_reranker.statistics import build_report, format_report  # noqa: E402


def load_model():
    try:
        model = build_reranker(RerankerConfig())
        print(f"loaded {RerankerConfig().backbone.model_name_or_path}")
        return model
    except Exception as exc:
        print(f"could not load the real checkpoint ({exc})")
        print("falling back to the tiny stub backbone.\n")

        config = RerankerConfig()
        config.backbone.model_name_or_path = "stub://tiny"
        config.backbone.hidden_size = 32
        config.backbone.embedding_dim = 48
        config.backbone.num_hidden_layers = 2
        config.fusion.fusion_dim = 32
        config.reranker_layers.hidden_dim = 32
        config.reranker_layers.intermediate_dim = 64
        config.heads.relevance_hidden_dim = 16
        config.heads.confidence_hidden_dim = 16
        config.validate()
        model = build_reranker(config)
        print("WARNING: the stub is untrained. Rankings below are meaningless;")
        print("         this example demonstrates the API, not model quality.\n")
        return model


def main() -> int:
    model = load_model()

    print(format_report(build_report(model, train_mode=model.config.training.train_mode)))
    print()

    inference = EmbeddingGemma2RerankerInference(model)

    response = inference.rerank(
        query="How do I configure authentication?",
        documents=[
            "Authentication is configured in the auth.yaml configuration file. "
            "Set the provider, client id and secret key there.",
            "The Pacific Ocean is the largest and deepest ocean on Earth.",
            "To configure authentication, mount a Kubernetes secret and reference "
            "it from your deployment manifest.",
            "Pasta carbonara requires guanciale, pecorino and eggs.",
        ],
    )

    print(f"Query: {response.query}")
    print()
    print("Results:")
    for item in response.results:
        marker = "  <<<" if item.rank == 1 else ""
        print(
            f"  rank={item.rank}  index={item.index}  "
            f"score={item.score:+.4f}  "
            f"calibrated={item.calibrated_score:.4f}  "
            f"confidence={item.confidence if item.confidence is not None else float('nan'):.4f}{marker}"
        )
        print(f"      {item.document[:78]}...")

    print()
    print(f"ranking               : {response.ranking}")
    print(f"top score             : {response.top_score:+.4f}")
    print(f"top1-top2 margin      : {response.margin:+.4f}")
    print(f"insufficient_confidence: {response.insufficient_confidence}")
    print(f"latency               : {response.metrics['latency_seconds'] * 1000:.1f} ms")
    print(f"documents/second      : {response.metrics['documents_per_second']:.1f}")

    print()
    print("JSON form:")
    print(json.dumps(response.to_dict(), indent=2, default=str)[:1200] + " ...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
