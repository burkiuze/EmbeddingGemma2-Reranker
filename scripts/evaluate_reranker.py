#!/usr/bin/env python3
"""Evaluate the reranker against the EmbeddingGemma cosine baseline.

    python scripts/evaluate_reranker.py --checkpoint runs/reranker \\
        --eval-file data/sample/eval.jsonl

    python scripts/evaluate_reranker.py --stub --eval-file data/sample/eval.jsonl

Reports both systems on identical inputs plus the delta. No improvement is
claimed until the numbers say so.
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
from evaluation.evaluate import (  # noqa: E402
    evaluate_and_write,
    format_report,
    load_checkpoint,
)
from training.dataset import load_jsonl, write_sample_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="training output directory")
    parser.add_argument("--eval-file", default=None, help="evaluation JSONL")
    parser.add_argument("--config", default=None, help="YAML config (used without --checkpoint)")
    parser.add_argument("--stub", action="store_true", help="use the tiny stub backbone")
    parser.add_argument("--ks", default="1,5,10", help="comma-separated cutoffs")
    parser.add_argument("--latency", action="store_true", help="measure latency/throughput")
    parser.add_argument("--no-baseline", action="store_true", help="skip the cosine baseline")
    parser.add_argument("--output", default=None, help="write metrics as JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ks = tuple(int(value) for value in args.ks.split(","))

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
        print(f"loaded checkpoint from {args.checkpoint}")
    elif args.config:
        model = build_reranker(RerankerConfig.from_yaml(args.config))
    elif args.stub:
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
        model = build_reranker(config.validate())
    else:
        raise SystemExit(
            "provide --checkpoint, --config or --stub to select a model"
        )

    if args.eval_file:
        examples = load_jsonl(args.eval_file)
        print(f"loaded {len(examples)} evaluation examples from {args.eval_file}")
    else:
        generated = write_sample_data("data/sample", num_examples=8)
        examples = load_jsonl(generated["eval"])
        print(f"no --eval-file given; using placeholder data at {generated['eval']}")
        print("WARNING: placeholder data yields meaningless metrics.")

    report = evaluate_and_write(
        model, examples, output_path=args.output, ks=ks, measure_latency=args.latency
    )

    print()
    print(format_report(report, ks=ks))

    if "deltas" in report:
        print()
        print("NOTE: deltas above compare the learned reranker to the cosine")
        print("      baseline. Negative values mean the baseline won.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
