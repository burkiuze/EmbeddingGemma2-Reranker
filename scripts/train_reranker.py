#!/usr/bin/env python3
"""CLI wrapper around :mod:`training.train`.

    python scripts/train_reranker.py --config configs/reranker.yaml \\
        --train-file data/train.jsonl --output-dir runs/exp1

    python scripts/train_reranker.py --smoke-test
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
