"""Configuration objects for EmbeddingGemma2-Reranker.

Everything the reranker needs is expressed as dataclasses so it can be built from
YAML, overridden from the CLI, and serialized back into a checkpoint directory
without any hidden global state.

The source-model fields (hidden size, layer count, vocabulary, tokenizer) mirror
``google/embeddinggemma-2`` exactly so the original checkpoint stays
load-compatible. Nothing in this file widens the backbone.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when a configuration value is inconsistent or unsupported."""


# Interaction mechanisms implemented by :mod:`embeddinggemma_reranker.interaction`.
SUPPORTED_INTERACTIONS: Tuple[str, ...] = (
    "none",
    "concatenate",
    "elementwise",
    "dot",
    "token",
)

# Pooling strategies implemented by :mod:`embeddinggemma_reranker.pooling`.
SUPPORTED_POOLING: Tuple[str, ...] = ("mean", "cls", "max", "last", "attention")

# Training modes understood by the trainer.
SUPPORTED_TRAIN_MODES: Tuple[str, ...] = ("head_only", "reranker_only", "lora", "full")

# Loss names understood by :mod:`training.losses`.
SUPPORTED_LOSSES: Tuple[str, ...] = (
    "bce",
    "pairwise_margin",
    "listwise",
    "mse",
    "distill_kl",
)

# Chunk aggregation strategies for long documents.
SUPPORTED_CHUNK_AGGREGATIONS: Tuple[str, ...] = ("max", "mean", "top_k_mean")


@dataclass
class PromptTemplates:
    """Asymmetric query/document prefixes.

    These match ``config_sentence_transformers.json`` in the upstream checkpoint
    so the reranker inherits the task conditioning EmbeddingGemma was trained with.
    """

    query: str = "task: search result | query: "
    document: str = "title: none | text: "
    enabled: bool = True

    def apply_query(self, text: str) -> str:
        return f"{self.query}{text}" if self.enabled else text

    def apply_document(self, text: str, title: Optional[str] = None) -> str:
        if not self.enabled:
            return text
        if title:
            return f"title: {title} | text: {text}"
        return f"{self.document}{text}"


@dataclass
class BackboneConfig:
    """How to load the EmbeddingGemma 2 text tower."""

    model_name_or_path: str = "google/embeddinggemma-2"
    dtype: str = "float32"
    device: str = "cpu"
    attn_implementation: Optional[str] = None

    # Verified upstream values. Overriding them is allowed for tiny test doubles
    # only; production should leave them alone so the checkpoint loads as-is.
    hidden_size: int = 512
    embedding_dim: int = 768
    num_hidden_layers: int = 24

    # Which transformer outputs feed the reranker. ``None`` means "last layer".
    hidden_layer_ids: Optional[Sequence[int]] = None
    load_vision_tower: bool = False
    load_audio_tower: bool = False

    def validate(self) -> "BackboneConfig":
        if self.hidden_size <= 0:
            raise ConfigError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.embedding_dim <= 0:
            raise ConfigError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.num_hidden_layers <= 0:
            raise ConfigError(
                f"num_hidden_layers must be positive, got {self.num_hidden_layers}"
            )
        if self.hidden_layer_ids is not None:
            for idx in self.hidden_layer_ids:
                if not isinstance(idx, int):
                    raise ConfigError(
                        f"hidden_layer_ids must be ints, got {type(idx).__name__}"
                    )
        return self


@dataclass
class FusionConfig:
    """Query/document feature fusion."""

    interaction_type: str = "elementwise"
    fusion_dim: int = 512
    dropout: float = 0.1
    layer_norm_features: bool = True
    activation: str = "gelu"

    def validate(self) -> "FusionConfig":
        if self.interaction_type not in SUPPORTED_INTERACTIONS:
            raise ConfigError(
                f"interaction_type must be one of {SUPPORTED_INTERACTIONS}, "
                f"got {self.interaction_type!r}"
            )
        if self.fusion_dim <= 0:
            raise ConfigError(f"fusion_dim must be positive, got {self.fusion_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        return self


@dataclass
class InteractionConfig:
    """Optional token-level interaction between query and document."""

    enabled: bool = False
    num_heads: int = 4
    projection_dim: int = 256
    max_query_tokens: int = 64
    max_document_tokens: int = 256
    dropout: float = 0.1
    use_max_pool: bool = True
    normalize_tokens: bool = True

    def validate(self) -> "InteractionConfig":
        if self.num_heads <= 0:
            raise ConfigError(f"num_heads must be positive, got {self.num_heads}")
        if self.projection_dim <= 0:
            raise ConfigError(
                f"projection_dim must be positive, got {self.projection_dim}"
            )
        if self.max_query_tokens <= 0 or self.max_document_tokens <= 0:
            raise ConfigError("token budgets must be positive")
        return self


@dataclass
class RerankerLayerConfig:
    """Shape of the trainable reranker blocks."""

    hidden_dim: int = 512
    num_layers: int = 2
    intermediate_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6

    def validate(self) -> "RerankerLayerConfig":
        if self.hidden_dim <= 0:
            raise ConfigError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 0:
            raise ConfigError(f"num_layers must be >= 0, got {self.num_layers}")
        if self.intermediate_dim <= 0:
            raise ConfigError(
                f"intermediate_dim must be positive, got {self.intermediate_dim}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        return self


@dataclass
class HeadConfig:
    """Relevance and confidence heads."""

    relevance_hidden_dim: int = 256
    relevance_dropout: float = 0.1
    confidence_head: bool = True
    confidence_hidden_dim: int = 256
    confidence_dropout: float = 0.1

    def validate(self) -> "HeadConfig":
        if self.relevance_hidden_dim <= 0:
            raise ConfigError("relevance_hidden_dim must be positive")
        if self.confidence_hidden_dim <= 0:
            raise ConfigError("confidence_hidden_dim must be positive")
        return self


@dataclass
class ChunkingConfig:
    """Long-document handling."""

    enabled: bool = True
    max_tokens: int = 512
    max_chunks: int = 8
    aggregation: str = "max"
    top_k: int = 3
    stride: int = 0

    def validate(self) -> "ChunkingConfig":
        if self.aggregation not in SUPPORTED_CHUNK_AGGREGATIONS:
            raise ConfigError(
                f"aggregation must be one of {SUPPORTED_CHUNK_AGGREGATIONS}, "
                f"got {self.aggregation!r}"
            )
        if self.max_tokens <= 0:
            raise ConfigError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_chunks <= 0:
            raise ConfigError(f"max_chunks must be positive, got {self.max_chunks}")
        if self.aggregation == "top_k_mean" and self.top_k <= 0:
            raise ConfigError(f"top_k must be positive, got {self.top_k}")
        if self.stride < 0:
            raise ConfigError(f"stride must be >= 0, got {self.stride}")
        return self


@dataclass
class PolicyConfig:
    """Abstention thresholds. Policy is never part of the neural forward pass."""

    minimum_relevance: Optional[float] = None
    minimum_margin: Optional[float] = None
    minimum_confidence: Optional[float] = None

    def validate(self) -> "PolicyConfig":
        for name in (
            "minimum_relevance",
            "minimum_margin",
            "minimum_confidence",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number or None, got {value!r}")
        return self


@dataclass
class TrainingConfig:
    """Optimization settings. Never triggers full fine-tuning on its own."""

    train_mode: str = "reranker_only"
    loss: str = "pairwise_margin"
    margin: float = 0.1
    listwise_temperature: float = 1.0
    distillation_weight: float = 0.0
    learning_rate: float = 2e-5
    head_learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42
    num_negatives: int = 1
    max_query_length: int = 64
    max_document_length: int = 256
    log_every: int = 10
    eval_every: int = 0
    save_every: int = 0
    output_dir: str = "runs/reranker"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[Sequence[str]] = None

    def validate(self) -> "TrainingConfig":
        if self.train_mode not in SUPPORTED_TRAIN_MODES:
            raise ConfigError(
                f"train_mode must be one of {SUPPORTED_TRAIN_MODES}, "
                f"got {self.train_mode!r}"
            )
        if self.loss not in SUPPORTED_LOSSES:
            raise ConfigError(
                f"loss must be one of {SUPPORTED_LOSSES}, got {self.loss!r}"
            )
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ConfigError(f"epochs must be positive, got {self.epochs}")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("gradient_accumulation_steps must be positive")
        if self.num_negatives <= 0:
            raise ConfigError("num_negatives must be positive")
        if self.distillation_weight < 0:
            raise ConfigError("distillation_weight must be >= 0")
        if self.train_mode == "full" and self.output_dir.strip() == "":
            raise ConfigError("full fine-tuning requires an explicit output_dir")
        return self


@dataclass
class RerankerConfig:
    """Top-level configuration for the whole reranker."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    reranker_layers: RerankerLayerConfig = field(default_factory=RerankerLayerConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prompts: PromptTemplates = field(default_factory=PromptTemplates)
    pooling: str = "mean"
    calibration: str = "sigmoid"

    def validate(self) -> "RerankerConfig":
        self.backbone.validate()
        self.fusion.validate()
        self.interaction.validate()
        self.reranker_layers.validate()
        self.heads.validate()
        self.chunking.validate()
        self.policy.validate()
        self.training.validate()
        if self.pooling not in SUPPORTED_POOLING:
            raise ConfigError(
                f"pooling must be one of {SUPPORTED_POOLING}, got {self.pooling!r}"
            )
        if self.calibration not in ("sigmoid", "none"):
            raise ConfigError(
                f"calibration must be 'sigmoid' or 'none', got {self.calibration!r}"
            )
        if self.interaction.enabled and self.fusion.interaction_type == "none":
            raise ConfigError(
                "interaction.enabled=True requires a non-'none' fusion.interaction_type"
            )
        return self

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerConfig":
        sections: Dict[str, Any] = {
            "backbone": BackboneConfig,
            "fusion": FusionConfig,
            "interaction": InteractionConfig,
            "reranker_layers": RerankerLayerConfig,
            "heads": HeadConfig,
            "chunking": ChunkingConfig,
            "policy": PolicyConfig,
            "training": TrainingConfig,
            "prompts": PromptTemplates,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in sections.items():
            payload = data.get(key)
            if payload is None:
                continue
            if not isinstance(payload, dict):
                raise ConfigError(f"config section {key!r} must be a mapping")
            known = {f.name for f in dataclasses.fields(klass)}
            unknown = set(payload) - known
            if unknown:
                raise ConfigError(
                    f"unknown keys in config section {key!r}: {sorted(unknown)}"
                )
            kwargs[key] = klass(**payload)
        for key in ("pooling", "calibration"):
            if key in data:
                kwargs[key] = data[key]
        unknown_top = set(data) - set(sections) - {"pooling", "calibration"}
        if unknown_top:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown_top)}")
        return cls(**kwargs).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "RerankerConfig":
        raw = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None):
        """Load YAML config, following an optional ``base:`` include chain."""
        if yaml is None:  # pragma: no cover
            raise ConfigError("PyYAML is required to load YAML configs")
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a YAML mapping")

        base = data.pop("base", None)
        if base:
            base_path = (path.parent / base).resolve()
            merged = cls._read_yaml_with_base(base_path)
            data = _deep_merge(merged, data)

        if overrides:
            data = _deep_merge(data, overrides)
        return cls.from_dict(data)

    @staticmethod
    def _read_yaml_with_base(path: Path) -> Dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = data.pop("base", None)
        if base:
            parent = cls._read_yaml_with_base((path.parent / base).resolve())
            data = _deep_merge(parent, data)
        return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; ``override`` wins on conflicts."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_cli_overrides(config: RerankerConfig, pairs: Sequence[str]) -> RerankerConfig:
    """Apply ``section.key=value`` CLI overrides onto an existing config."""
    nested: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override must look like section.key=value, got {pair!r}")
        dotted, _, raw = pair.partition("=")
        parts = dotted.split(".")
        if len(parts) < 2:
            raise ConfigError(f"override must look like section.key=value, got {pair!r}")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - defensive
                raise ConfigError(f"conflicting override path in {pair!r}")
        cursor[parts[-1]] = _coerce(raw)

    merged = _deep_merge(config.to_dict(), nested)
    return RerankerConfig.from_dict(merged)


def _coerce(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce(item.strip()) for item in inner.split(",")]
    return raw


def config_summary(config: RerankerConfig) -> str:
    """Human-readable one-block summary used by the CLI entry points."""
    lines: List[str] = [
        "EmbeddingGemma2-Reranker configuration",
        f"  backbone            : {config.backbone.model_name_or_path}"
        f" (hidden={config.backbone.hidden_size}, embed={config.backbone.embedding_dim})",
        f"  interaction_type    : {config.fusion.interaction_type}",
        f"  token interaction   : {'on' if config.interaction.enabled else 'off'}",
        f"  reranker_dim        : {config.reranker_layers.hidden_dim}",
        f"  num_reranker_layers : {config.reranker_layers.num_layers}",
        f"  confidence head     : {'on' if config.heads.confidence_head else 'off'}",
        f"  chunking            : {config.chunking.aggregation} @ {config.chunking.max_tokens} tokens",
        f"  pooling             : {config.pooling}",
        f"  prompts             : {'on' if config.prompts.enabled else 'off'}",
        f"  train_mode          : {config.training.train_mode}",
        f"  loss                : {config.training.loss}",
    ]
    return "\n".join(lines)
