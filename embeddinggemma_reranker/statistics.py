"""Parameter statistics.

Every number printed here is counted from real tensors via ``numel()``. Nothing is
estimated from config values, and nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


def count_parameters(module: Optional[nn.Module]) -> int:
    """Total number of parameter elements in ``module``."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def count_trainable(module: Optional[nn.Module]) -> int:
    """Number of parameter elements with ``requires_grad=True``."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_frozen(module: Optional[nn.Module]) -> int:
    return count_parameters(module) - count_trainable(module)


def freeze(module: Optional[nn.Module], frozen: bool = True) -> int:
    """Set ``requires_grad`` on every parameter; returns how many changed."""
    if module is None:
        return 0
    changed = 0
    for param in module.parameters():
        if param.requires_grad != (not frozen):
            param.requires_grad = not frozen
            changed += param.numel()
    return changed


def breakdown(model: nn.Module) -> Dict[str, int]:
    """Per-component parameter counts for the reranker."""
    from .model import EmbeddingGemma2Reranker

    sections: Dict[str, nn.Module] = {
        "backbone": getattr(model, "backbone", None),
        "fusion": getattr(model, "fusion", None),
        "interaction": getattr(model, "token_interaction", None),
        "reranker_layers": getattr(model, "reranker", None),
        "relevance_head": getattr(model, "relevance_head", None),
        "confidence_head": getattr(model, "confidence_head", None),
    }

    stats = {name: count_parameters(module) for name, module in sections.items()}

    backbone_params = stats["backbone"]
    added = sum(
        value for name, value in stats.items() if name != "backbone"
    )
    stats["added_parameters"] = added
    stats["total_parameters"] = backbone_params + added
    stats["trainable_parameters"] = count_trainable(model)
    stats["frozen_parameters"] = count_frozen(model)

    total = stats["total_parameters"]
    trainable = stats["trainable_parameters"]
    stats["trainable_percentage"] = (
        round(100.0 * trainable / total, 4) if total else 0.0
    )
    stats["added_vs_backbone_ratio"] = (
        round(added / backbone_params, 4) if backbone_params else 0.0
    )
    return stats


def format_bytes(num_bytes: float) -> str:
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < step:
            return f"{value:.2f} {unit}"
        value /= step
    return f"{value:.2f} TiB"


def estimate_memory(model: nn.Module) -> Dict[str, Any]:
    """Approximate resident size from parameter dtypes and buffers."""
    param_bytes = 0
    buffer_bytes = 0
    for tensor in model.parameters():
        param_bytes += tensor.numel() * tensor.element_size()
    for tensor in model.buffers():
        buffer_bytes += tensor.numel() * tensor.element_size()

    trainable = count_trainable(model)
    gradient_bytes = trainable * 4  # float32 gradients in the common case
    total = param_bytes + buffer_bytes

    return {
        "parameter_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "gradient_bytes_estimate": gradient_bytes,
        "total_bytes": total,
        "parameter_memory": format_bytes(param_bytes),
        "total_memory": format_bytes(total),
    }


@dataclass
class ParameterReport:
    """Structured result of :func:`build_report`."""

    counts: Dict[str, int] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    train_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts,
            "memory": self.memory,
            "train_mode": self.train_mode,
        }


def build_report(model: nn.Module, train_mode: Optional[str] = None) -> ParameterReport:
    return ParameterReport(
        counts=breakdown(model),
        memory=estimate_memory(model),
        train_mode=train_mode,
    )


def format_report(report: ParameterReport) -> str:
    counts = report.counts
    lines = [
        "Parameter statistics (counted from tensors)",
        f"  EmbeddingGemma backbone : {counts['backbone']:>12,}",
        f"  Fusion                   : {counts['fusion']:>12,}",
        f"  Interaction              : {counts['interaction']:>12,}",
        f"  Reranker layers          : {counts['reranker_layers']:>12,}",
        f"  Relevance head           : {counts['relevance_head']:>12,}",
        f"  Confidence head          : {counts['confidence_head']:>12,}",
        f"  {'-' * 12}",
        f"  Total added parameters   : {counts['added_parameters']:>12,}",
        f"  Total parameters         : {counts['total_parameters']:>12,}",
        f"  Trainable parameters     : {counts['trainable_parameters']:>12,}",
        f"  Frozen parameters        : {counts['frozen_parameters']:>12,}",
        f"  Trainable percentage     : {counts['trainable_percentage']:>11}%",
        f"  Added / backbone ratio   : {counts['added_vs_backbone_ratio']:>11}x",
        "",
        f"  Parameter memory         : {report.memory.get('parameter_memory', 'n/a')}",
        f"  Total memory (w/ buffers): {report.memory.get('total_memory', 'n/a')}",
    ]
    if report.train_mode:
        lines.insert(1, f"  train_mode                : {report.train_mode}")
    return "\n".join(lines)


def trainable_parameter_list(model: nn.Module) -> Iterable[str]:
    """Names of parameters that will receive gradients."""
    return (name for name, param in model.named_parameters() if param.requires_grad)
