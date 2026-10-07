#!/usr/bin/env python3
"""Inspect the real EmbeddingGemma 2 source architecture.

Prints what is actually present — config, module tree, tensor shapes, projection
modules, pooling configuration, parameter counts — rather than reciting README
values. When the real checkpoint is unavailable it falls back to the local config
file and says clearly that no weights were loaded.

    python scripts/inspect_source.py
    python scripts/inspect_source.py --model google/embeddinggemma-2
    python scripts/inspect_source.py --config-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "google/embeddinggemma-2"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_config_file(model_name: str) -> Optional[Dict[str, Any]]:
    """Read the checkpoint's config.json without downloading weights."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed; skipping config fetch.")
        return None

    try:
        path = hf_hub_download(model_name, "config.json")
    except Exception as exc:
        print(f"could not fetch config.json for {model_name}: {exc}")
        return None

    return json.loads(Path(path).read_text(encoding="utf-8"))


def report_config(config: Dict[str, Any]) -> None:
    section("Checkpoint config (config.json)")

    print(f"model_type      : {config.get('model_type')}")
    print(f"architectures   : {config.get('architectures')}")
    print(f"dtype           : {config.get('dtype')}")

    text = config.get("text_config", {})
    if not text:
        print("no text_config present")
        return

    print("\n--- text_config (the 270M target) ---")
    for key in (
        "hidden_size",
        "embedding_dim",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "hidden_activation",
        "vocab_size",
        "rms_norm_eps",
        "sliding_window",
        "hidden_size_per_layer_input",
        "max_position_embeddings",
    ):
        if key in text:
            print(f"  {key:<28}: {text[key]}")

    layer_types = text.get("layer_types")
    if layer_types:
        counts: Dict[str, int] = {}
        for kind in layer_types:
            counts[kind] = counts.get(kind, 0) + 1
        print(f"  {'layer_types':<28}: {len(layer_types)} layers {counts}")

    per_layer = text.get("per_layer_config")
    if per_layer:
        print(f"  {'per_layer_config':<28}: {per_layer}")

    rope = text.get("rope_parameters")
    if rope:
        print(f"  {'rope_parameters':<28}: {rope}")

    for modality in ("vision_config", "audio_config"):
        if modality in config and config[modality]:
            other = config[modality]
            print(
                f"\n  {modality}: model_type={other.get('model_type')}, "
                f"layers={other.get('num_hidden_layers')}, "
                f"hidden={other.get('hidden_size')}"
            )
            print(
                "    (not loaded for the text-only 270M target)"
            )


def report_pooling(model_name: str) -> None:
    section("Pooling / SentenceTransformers metadata")

    for filename in ("1_Pooling/config.json", "config_sentence_transformers.json"):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name, filename)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"{filename}: unavailable ({exc})")
            continue

        print(f"\n--- {filename} ---")
        if filename.startswith("1_Pooling"):
            for key, value in data.items():
                print(f"  {key:<24}: {value}")
        else:
            print(f"  similarity_fn_name    : {data.get('similarity_fn_name')}")
            print(f"  default_prompt_name   : {data.get('default_prompt_name')}")
            prompts = data.get("prompts", {})
            for key in ("query", "document", "Reranking", "CodeRetrieval"):
                if key in prompts:
                    print(f"  prompt {key:<15}: {prompts[key]!r}")


def report_module_tree(model: Any) -> None:
    section("Module tree (top levels)")

    for name, module in model.named_children():
        children = list(module.named_children())
        print(f"{name}")
        if not children:
            print("  (leaf)")
            continue
        for child_name, child in children[:12]:
            kind = type(child).__name__
            count = sum(1 for _ in child.parameters())
            print(f"  - {child_name:<28} {kind:<28} params={count:,}")
        if len(children) > 12:
            print(f"  ... {len(children) - 12} more")


def report_tensor_shapes(model: Any, limit: int = 60) -> None:
    section("Tensor shapes (largest parameters)")

    named = [(n, p) for n, p in model.named_parameters()]
    named.sort(key=lambda item: item[1].numel(), reverse=True)

    for name, param in named[:limit]:
        print(f"  {name:<62} {str(tuple(param.shape)):<22} {param.numel():>12,}")
    if len(named) > limit:
        print(f"  ... {len(named) - limit} more parameters")


def report_projections(model: Any) -> None:
    section("Projection modules (hidden -> embedding dim)")

    import torch.nn as nn

    found = False
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features and out_features and in_features != out_features:
            print(f"  {name:<58} {in_features} -> {out_features}")
            found = True
    if not found:
        print("  no width-changing Linear modules found at this depth")


def report_parameter_totals(model: Any) -> None:
    section("Parameter totals")

    total = sum(p.numel() for p in model.parameters())
    dtype_counts: Dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype).replace("torch.", "")
        dtype_counts[key] = dtype_counts.get(key, 0) + param.numel()

    print(f"  total parameters : {total:,}")
    for dtype, count in sorted(dtype_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {dtype:<16}: {count:,} ({100 * count / total:.1f}%)")
    print(f"  approx fp32 size : {total * 4 / 1024 ** 2:.1f} MiB")
    print(f"  approx bf16 size : {total * 2 / 1024 ** 2:.1f} MiB")


def load_model(model_name: str) -> Optional[Any]:
    try:
        from transformers import AutoModel
    except ImportError:
        print("transformers is not installed; cannot load weights.")
        return None

    try:
        import torch

        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, dtype=torch.bfloat16
        )
    except Exception as exc:
        print(f"could not load {model_name}: {exc}")
        print("This is expected without network access or the checkpoint on disk.")
        return None
    return model


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="skip weight loading; inspect config and pooling metadata only",
    )
    parser.add_argument("--output", default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    print("EmbeddingGemma 2 source inspection")
    print(f"model: {args.model}")

    section("Local environment")
    print(f"  python : {sys.version.split()[0]}")
    try:
        import torch

        print(f"  torch  : {torch.__version__}")
    except ImportError:
        print("  torch  : not installed")
    try:
        import transformers

        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        print("  transformers: not installed")

    config = load_config_file(args.model)
    if config:
        report_config(config)
    report_pooling(args.model)

    model = None
    if not args.config_only:
        model = load_model(args.model)
        if model is not None:
            report_module_tree(model)
            report_projections(model)
            report_tensor_shapes(model)
            report_parameter_totals(model)
        else:
            section("Weights")
            print("  No weights were loaded. Parameter totals above are therefore")
            print("  unavailable; only the file-based metadata is reported.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "weights_loaded": model is not None,
            "config": config,
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nreport written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
