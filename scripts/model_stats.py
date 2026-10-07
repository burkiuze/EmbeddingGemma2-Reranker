#!/usr/bin/env python3
"""Print real parameter statistics for the reranker.

Every number is counted from tensors with ``numel()``. Nothing is estimated.

    python scripts/model_stats.py
    python scripts/model_stats.py --config configs/reranker.yaml
    python scripts/model_stats.py --train-mode reranker_only
    python scripts/model_stats.py --component-model backbone
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddinggemma_reranker.config import RerankerConfig  # noqa: E402
from embeddinggemma_reranker.model import build_reranker  # noqa: E402
from embeddinggemma_reranker.statistics import (  # noqa: E402
    build_report,
    format_report,
)
from training.trainer import configure_trainable  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--component-model",
        default=None,
        help="override backbone path (e.g. a small local checkpoint for testing)",
    )
    parser.add_argument(
        "--train-mode",
        default=None,
        choices=["head_only", "reranker_only", "lora", "full"],
        help="freeze according to this mode before reporting",
    )
    parser.add_argument("--stub", action="store_true", help="use the tiny stub backbone")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    config = RerankerConfig.from_yaml(args.config) if args.config else RerankerConfig()

    if args.stub or args.component_model:
        config.backbone.model_name_or_path = args.component_model or "stub://tiny"
        config.backbone.hidden_size = 32
        config.backbone.embedding_dim = 48
        config.backbone.num_hidden_layers = 2
        config.fusion.fusion_dim = 32
        config.reranker_layers.hidden_dim = 32
        config.reranker_layers.intermediate_dim = 64
        config.heads.relevance_hidden_dim = 16
        config.heads.confidence_hidden_dim = 16
    config.validate()

    print(f"backbone: {config.backbone.model_name_or_path}")
    print(
        f"architecture: interaction_type={config.fusion.interaction_type}, "
        f"token_interaction={'on' if config.interaction.enabled else 'off'}, "
        f"reranker_dim={config.reranker_layers.hidden_dim}, "
        f"num_reranker_layers={config.reranker_layers.num_layers}"
    )
    if args.stub or args.component_model:
        print("NOTE: stub backbone — parameter counts below are NOT the real model.")

    print()
    model = build_reranker(config)

    if args.train_mode:
        configure_trainable(model, args.train_mode)
        print(f"train_mode: {args.train_mode}")
        print()

    report = build_report(model, train_mode=args.train_mode)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwritten to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
