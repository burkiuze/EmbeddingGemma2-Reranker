#!/usr/bin/env python3
"""Rerank code-search candidates with a natural-language query.

EmbeddingGemma 2 supports code natively, so no code-specific tokenizer is
introduced. Code is treated as text with the `CodeRetrieval` prompt convention.

    python examples/code_rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddinggemma_reranker.config import (  # noqa: E402
    PromptTemplates,
    RerankerConfig,
)
from embeddinggemma_reranker.inference import EmbeddingGemma2RerankerInference  # noqa: E402
from embeddinggemma_reranker.model import build_reranker  # noqa: E402

CANDIDATES = [
    {
        "id": "auth/jwt.py::decode_token",
        "title": "auth/jwt.py",
        "text": (
            "def decode_token(token: str) -> dict:\n"
            "    payload = jwt.decode(token, options={'verify_signature': False})\n"
            "    return json.loads(payload)"
        ),
    },
    {
        "id": "auth/jwt.py::validate_expiration",
        "title": "auth/jwt.py",
        "text": (
            "def validate_expiration(token: str) -> bool:\n"
            "    claims = jwt.decode(token, verify_signature=False)\n"
            "    return claims['exp'] > time.time()"
        ),
    },
    {
        "id": "auth/session.py::set_cookie",
        "title": "auth/session.py",
        "text": (
            "def set_cookie(response, name, value, max_age=3600):\n"
            "    response.set_cookie(name, value, httponly=True, samesite='Lax')\n"
            "    return response"
        ),
    },
    {
        "id": "db/pool.py::connect",
        "title": "db/pool.py",
        "text": (
            "def connect(dsn: str, pool_size: int = 10):\n"
            "    return psycopg2.pool.ThreadedConnectionPool(1, pool_size, dsn)"
        ),
    },
    {
        "id": "util/time.py::humanize",
        "title": "util/time.py",
        "text": (
            "def humanize(seconds: int) -> str:\n"
            "    for unit in ('s', 'm', 'h', 'd'):\n"
            "        if seconds < 60:\n"
            "            return f'{seconds:.0f}{unit}'\n"
            "        seconds /= 60"
        ),
    },
]


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
        config.prompts = PromptTemplates(
            query="task: code retrieval | query: ",
            document="title: none | text: ",
            enabled=True,
        )
        print("using the tiny stub backbone — rankings are meaningless.\n")
        return build_reranker(config.validate()), True


def main() -> int:
    model, is_stub = load_model()
    print("WARNING: reranker heads are untrained; ranking quality has not been evaluated.")

    # Use the upstream code-retrieval prompt on both sides.
    model.config.prompts.query = "task: code retrieval | query: "
    model.config.prompts.enabled = True

    inference = EmbeddingGemma2RerankerInference(model)

    queries = [
        "function that validates JWT expiration",
        "how to rotate an auth cookie",
        "open a database connection pool",
    ]

    for query in queries:
        print("=" * 74)
        print(f"Query: {query}")
        print("=" * 74)

        response = inference.rerank(
            query,
            [candidate["text"] for candidate in CANDIDATES],
            document_ids=[candidate["id"] for candidate in CANDIDATES],
        )

        for item in response.results:
            print(f"  rank={item.rank}  {item.index}")
            first_line = item.document.splitlines()[0]
            print(f"      {first_line}")

        print(f"\n  ranking : {response.ranking}")
        print(f"  top score: {response.top_score:+.4f}")

        if is_stub:
            print("  (stub model: pipeline demonstration only)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
