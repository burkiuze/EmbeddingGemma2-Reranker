#!/usr/bin/env python3
"""Two-stage retrieval: bi-encoder first stage, learned reranker second stage.

Demonstrates the actual reason to build a reranker at all. The bi-encoder
(EmbeddingGemma cosine similarity) retrieves cheaply; the cross-encoder then
re-scores that candidate list with full query/document interaction.

    python examples/semantic_search_rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from embeddinggemma_reranker.config import RerankerConfig  # noqa: E402
from embeddinggemma_reranker.inference import EmbeddingGemma2RerankerInference  # noqa: E402
from embeddinggemma_reranker.model import build_reranker  # noqa: E402
from evaluation.baseline import BiEncoderBaseline  # noqa: E402

CORPUS = [
    "Password reset: open Settings, choose Security, then Send a reset link to your email.",
    "Two-factor authentication can be enabled under Account settings > Security.",
    "Pasta carbonara: guanciale, pecorino romano, egg yolks, black pepper. No cream.",
    "OAuth 2.0 client credentials flow: the service authenticates with a token endpoint.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Deployments are promoted by building an artifact, verifying tests, then releasing.",
    "Session cookies are signed with a secret key and expire after a configured TTL.",
    "PostgreSQL connection strings use the postgresql://user:pass@host:5432/db form.",
    "Rainbows form when light refracts inside water droplets and disperses by wavelength.",
    "To rotate an API key, create a new key, deploy it, then revoke the previous one.",
]

RELEVANCE = {
    "How do I reset a password?": {0: 3.0, 5: 0.0},
    "How do I enable two-factor authentication?": {1: 3.0, 0: 1.0, 6: 0.0},
    "How do I connect to PostgreSQL?": {7: 3.0, 3: 1.0, 0: 0.0},
    "How do I rotate an API key?": {9: 3.0, 0: 0.0, 6: 0.0},
}


def load_model():
    try:
        return build_reranker(RerankerConfig()), False
    except Exception as exc:
        print(f"could not load the real checkpoint ({exc})")
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
        print("using the tiny stub backbone — rankings are meaningless, the")
        print("two-stage *plumbing* is what this example demonstrates.\n")
        return build_reranker(config.validate()), True


def main() -> int:
    model, is_stub = load_model()
    baseline = BiEncoderBaseline(model)
    reranker = EmbeddingGemma2RerankerInference(model)

    top_k = 4

    for query, relevant in RELEVANCE.items():
        print("=" * 74)
        print(f"Query: {query}")
        print("=" * 74)

        # ---- Stage 1: bi-encoder retrieval over the whole corpus
        first_stage = baseline.rank(query, CORPUS)[:top_k]
        print(f"\nStage 1 — EmbeddingGemma cosine, top {top_k} of {len(CORPUS)}:")
        for rank, index in enumerate(first_stage, start=1):
            mark = "  [relevant]" if index in relevant else ""
            print(f"  {rank}. (corpus idx {index}) {CORPUS[index][:58]}...{mark}")

        # ---- Stage 2: learned reranker over those candidates only
        candidates = [CORPUS[index] for index in first_stage]
        response = reranker.rerank(query, candidates)

        print(f"\nStage 2 — learned reranker re-scored {len(candidates)} candidates:")
        for item in response.results:
            corpus_index = first_stage[item.index]
            mark = "  [relevant]" if corpus_index in relevant else ""
            print(
                f"  rank={item.rank}  corpus idx={corpus_index}  "
                f"score={item.score:+.4f}{mark}"
            )

        if is_stub:
            print("\n  (stub model: no ranking quality to interpret)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
