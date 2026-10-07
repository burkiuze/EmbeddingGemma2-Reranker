"""Training entry point.

    python -m training.train --config configs/reranker.yaml \\
        --train-file data/train.jsonl --output-dir runs/exp1

``--smoke-test`` runs a full training step against a tiny stub backbone so the
pipeline can be validated without downloading any weights. A smoke test that
passes says nothing about model quality; it only says the code runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from embeddinggemma_reranker.config import (
    RerankerConfig,
    apply_cli_overrides,
    config_summary,
)
from embeddinggemma_reranker.model import build_reranker

from .dataset import RerankingExample, load_jsonl, split_examples, write_sample_data
from .hard_negatives import MiningConfig, load_corpus, mine_hard_negatives, write_mined
from .trainer import FullFinetuneNotConfirmed, RerankerTrainer, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training.train",
        description="Train the EmbeddingGemma 2 reranker.",
    )
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--train-file", default=None, help="training JSONL")
    parser.add_argument("--eval-file", default=None, help="validation JSONL")
    parser.add_argument("--output-dir", default=None, help="checkpoint directory")
    parser.add_argument("--corpus-file", default=None, help="corpus JSONL for mining")
    parser.add_argument(
        "--mine-hard-negatives",
        action="store_true",
        help="mine hard negatives with the bi-encoder before training",
    )
    parser.add_argument("--num-hard-negatives", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--allow-full-finetune",
        action="store_true",
        help="required acknowledgement for --set training.train_mode=full",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one training step on a tiny stub backbone",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="config override; repeatable",
    )
    return parser


def load_config(args: argparse.Namespace) -> RerankerConfig:
    if args.config:
        config = RerankerConfig.from_yaml(args.config)
    else:
        config = RerankerConfig()

    if args.overrides:
        config = apply_cli_overrides(config, args.overrides)

    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.num_hard_negatives is not None:
        config.training.num_negatives = args.num_hard_negatives
    if args.config is None or not args.overrides:
        config.validate()
    return config


def smoke_examples() -> List[RerankingExample]:
    """Three tiny examples. Deliberately trivial: this exercises code paths."""
    return [
        RerankingExample(
            query="How do I reset a password?",
            documents=[
                "Password reset instructions: open account settings.",
                "How to prepare pasta: boil water, add salt.",
            ],
            relevance=[2.0, 0.0],
        ),
        RerankingExample(
            query="What is Python?",
            documents=[
                "Python is a programming language.",
                "The Pacific Ocean is the largest ocean.",
            ],
            relevance=[2.0, 0.0],
        ),
        RerankingExample(
            query="JWT expiration check",
            documents=[
                "def validate_jwt(token): return token.expired",
                "def bake_cake(): preheat oven.",
                "The sky appears blue due to Rayleigh scattering.",
            ],
            relevance=[3.0, 0.0, 0.0],
        ),
    ]


def run_smoke_test() -> int:
    """End-to-end check on a stub backbone: build, train, save, reload."""
    print("=== smoke test (stub backbone, no weights downloaded) ===")
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
    config.training.max_query_length = 24
    config.training.max_document_length = 32
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.output_dir = "runs/smoke"
    config.validate()

    model = build_reranker(config)
    print(config_summary(config))

    from embeddinggemma_reranker.statistics import build_report, format_report

    print()
    print(format_report(build_report(model, config.training.train_mode)))

    trainer = RerankerTrainer(model, config)
    summary = trainer.train(smoke_examples())
    print(f"\ntrained parameters : {summary.trainable_parameters:,}")
    print(f"total parameters   : {summary.total_parameters:,}")
    print(f"steps completed    : {summary.steps_completed}/{summary.steps}")
    print(f"final loss         : {summary.final_loss}")

    output_dir = trainer.save(summary)
    print(f"checkpoint written : {output_dir}")

    # The saved config must round-trip.
    reloaded = RerankerConfig.from_json(output_dir / "reranker_config.json")
    print(f"config round-trip  : ok ({reloaded.fusion.interaction_type})")

    # And inference must work on the same example from the brief.
    from embeddinggemma_reranker.inference import EmbeddingGemma2RerankerInference

    inference = EmbeddingGemma2RerankerInference(model, config)
    response = inference.rerank(
        query="How do I reset a password?",
        documents=[
            "Password reset instructions: open account settings.",
            "How to prepare pasta: boil water, add salt.",
            "Account security and login help.",
        ],
    )
    print(f"\nsmoke rerank ranking: {response.ranking}")
    print(f"num results         : {len(response.results)}")
    print(f"all documents scored: {len(response.results) == 3}")
    print("\nNOTE: an untrained stub has no ranking quality. This validates plumbing only.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.smoke_test:
        return run_smoke_test()

    config = load_config(args)
    set_seed(config.training.seed)

    train_path = args.train_file
    if train_path is None:
        # No dataset supplied: write a tiny placeholder so the command still runs.
        sample_dir = Path(config.training.output_dir).parent / "sample_data"
        generated = write_sample_data(sample_dir, num_examples=8)
        train_path = str(generated["train"])
        print(f"no --train-file given; using generated placeholder data at {train_path}")
        print("WARNING: placeholder data cannot produce a usable model.")

    examples = load_jsonl(train_path, num_negatives=config.training.num_negatives)
    print(f"loaded {len(examples)} examples from {train_path}")

    if args.mine_hard_negatives:
        if not args.corpus_file:
            raise SystemExit(
                "--mine-hard-negatives requires --corpus-file (a retrieval corpus JSONL)"
            )
        corpus = load_corpus(args.corpus_file)
        model = build_reranker(config)
        mining_config = MiningConfig(
            top_k=args.top_k or max(4, config.training.num_negatives * 4),
            num_hard_negatives=config.training.num_negatives,
        )
        examples, stats = mine_hard_negatives(model, examples, corpus, mining_config)
        print(f"mining stats: {json.dumps(stats.to_dict(), indent=2)}")
        mined_path = Path(config.training.output_dir) / "mined_train.jsonl"
        write_mined(mined_path, examples, stats)
        print(f"mined data written to {mined_path}")

    if args.eval_file:
        validation = load_jsonl(args.eval_file)
        print(f"loaded {len(validation)} validation examples")
    else:
        train, validation = split_examples(
            examples, args.validation_ratio, config.training.seed
        )
        examples = train
        print(f"split: {len(train)} train / {len(validation)} validation")

    model = build_reranker(config)
    trainer = RerankerTrainer(model, config)

    from embeddinggemma_reranker.statistics import build_report, format_report

    print()
    print(format_report(build_report(model, config.training.train_mode)))

    try:
        summary = trainer.train(examples, allow_full_finetune=args.allow_full_finetune)
    except FullFinetuneNotConfirmed as exc:
        print(f"\nrefused to train: {exc}", file=sys.stderr)
        return 2

    output_dir = trainer.save(summary)
    print(f"\ncheckpoint written to {output_dir}")
    print(json.dumps(summary.to_dict(), indent=2))

    if validation:
        from evaluation.evaluate import evaluate_reranker

        metrics = evaluate_reranker(model, validation, ks=(1, 5, 10))
        print("\nvalidation metrics:")
        print(json.dumps(metrics["reranker"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
