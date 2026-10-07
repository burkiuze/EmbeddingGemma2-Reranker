"""EmbeddingGemma 2 text backbone adapter.

Wraps the upstream ``EmbeddingGemma2Model`` text tower and exposes three things a
reranker needs that a plain embedding call does not give you:

* **token hidden states** from any transformer layer (not just the final 768-d vector),
* the **native 768-d embedding** so the bi-encoder baseline stays exact,
* a clean seam for a tiny stub in tests so unit tests never download weights.

The original checkpoint is never mutated. ``hidden_size``, layer count, vocabulary
and tokenizer all come straight from ``google/embeddinggemma-2``'s ``config.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BackboneConfig, ConfigError
from .pooling import pool


@dataclass
class BackboneOutput:
    """Everything the reranker needs from one backbone call."""

    #: ``(B, T, H)`` token-level states from the selected layer(s).
    token_hidden: torch.Tensor
    #: ``(B, H)`` pooled sequence representation.
    pooled: torch.Tensor
    #: ``(B, E)`` native normalized embedding (``E`` = 768 upstream).
    embedding: Optional[torch.Tensor]
    #: ``(B, T)`` boolean mask of attended positions.
    attention_mask: torch.Tensor
    #: Number of transformer layers the backbone exposes, when known.
    num_hidden_layers: Optional[int] = None

    def to(self, device: torch.device) -> "BackboneOutput":
        return BackboneOutput(
            token_hidden=self.token_hidden.to(device),
            pooled=self.pooled.to(device),
            embedding=None if self.embedding is None else self.embedding.to(device),
            attention_mask=self.attention_mask.to(device),
            num_hidden_layers=self.num_hidden_layers,
        )


class EmbeddingGemma2TextBackbone(nn.Module):
    """Shared EmbeddingGemma 2 text encoder.

    One instance is shared by :class:`~embeddinggemma_reranker.model.QueryEncoder`
    and :class:`~embeddinggemma_reranker.model.DocumentEncoder` by default, so a
    270M tower is loaded into memory exactly once.
    """

    def __init__(
        self,
        config: Optional[BackboneConfig] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = (config or BackboneConfig()).validate()
        self._stub = model is None and self.config.model_name_or_path.startswith(
            ("stub://", "tiny://", "mock://")
        )

        if self._stub:
            self.model = _TinyTextTower(
                hidden_size=self.config.hidden_size,
                embedding_dim=self.config.embedding_dim,
                num_hidden_layers=self.config.num_hidden_layers,
            )
            self.tokenizer = tokenizer or _StubTokenizer(self.config.vocab_size_guess)
            self.num_hidden_layers = self.config.num_hidden_layers
            self.hidden_size = self.config.hidden_size
            self.embedding_dim = self.config.embedding_dim
            return

        self.model = model if model is not None else self._load_from_pretrained()
        self.tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._read_module_shapes()

    # ------------------------------------------------------------- loading
    def _load_from_pretrained(self) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "transformers is required to load EmbeddingGemma 2. "
                "Install it with `pip install transformers`."
            ) from exc

        kwargs: Dict[str, Any] = {
            "dtype": getattr(torch, self.config.dtype, torch.float32),
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        model = AutoModel.from_pretrained(
            self.config.model_name_or_path,
            output_hidden_states=True,
            **kwargs,
        )
        if not self.config.load_vision_tower:
            model = _disable_modality_towers(model)
        return model

    def _load_tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "transformers is required to load the EmbeddingGemma tokenizer."
            ) from exc
        return AutoTokenizer.from_pretrained(self.config.model_name_or_path)

    def _read_module_shapes(self) -> None:
        text_config = getattr(self.model, "config", None)
        text_config = getattr(text_config, "text_config", text_config)

        self.hidden_size = int(
            getattr(text_config, "hidden_size", self.config.hidden_size)
        )
        self.embedding_dim = int(
            getattr(text_config, "embedding_dim", self.config.embedding_dim)
        )
        self.num_hidden_layers = int(
            getattr(text_config, "num_hidden_layers", self.config.num_hidden_layers)
        )

    # -------------------------------------------------------------- helpers
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless stubs
            return torch.device("cpu")

    def _select_layers(
        self, hidden_states: Optional[Tuple[torch.Tensor, ...]], batch: int, seq: int
    ) -> torch.Tensor:
        """Pick the configured transformer layer(s) and concatenate them."""
        if not hidden_states:
            # Backbone did not expose hidden states: fall back to the pooled
            # output expanded per-token so downstream shapes still hold.
            raise ConfigError(
                "backbone returned no hidden states; the reranker needs at least "
                "one transformer layer output"
            )

        # Transformers appends the final projected 768-d output to the captured
        # 512-d transformer states. Fusion consumes pre-projection token states.
        hidden_states = tuple(state for state in hidden_states if state.shape[-1] == self.hidden_size)
        if not hidden_states:
            raise ConfigError("backbone exposed no pre-projection token hidden states")
        requested = self.config.hidden_layer_ids
        if requested is None:
            chosen = [len(hidden_states) - 1]
        else:
            chosen = [i if i >= 0 else len(hidden_states) + i for i in requested]

        valid = [i for i in chosen if 0 <= i < len(hidden_states)]
        if not valid:
            raise ConfigError(
                f"hidden_layer_ids={list(requested)} are out of range for "
                f"{len(hidden_states)} available hidden-state tuples"
            )

        if len(valid) == 1:
            return hidden_states[valid[0]]

        width = sum(hidden_states[i].shape[-1] for i in valid)
        parts = [hidden_states[i] for i in valid]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden: bool = True,
    ) -> BackboneOutput:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden,
        }
        # Only pass token_type_ids when the backbone actually declares them;
        # Gemma-style text towers do not accept it.
        if getattr(self.model, "config", None) is not None and hasattr(
            self.model.config, "type_vocab_size"
        ):
            kwargs["token_type_ids"] = torch.zeros_like(input_ids)

        raw = self.model(**kwargs)

        mask_bool = attention_mask.to(torch.bool)
        if not bool(mask_bool.any(dim=-1).all()):
            raise ValueError("every sequence must contain at least one attended token")

        pooled_native = self._native_embedding(raw, attention_mask)
        if output_hidden:
            hidden_tuple = getattr(raw, "hidden_states", None)
            if hidden_tuple is None:
                token_hidden = pooled_native.unsqueeze(1).expand(
                    input_ids.shape[0], input_ids.shape[1], pooled_native.shape[-1]
                )
            else:
                token_hidden = self._select_layers(hidden_tuple, input_ids.shape[0], input_ids.shape[1])
        else:
            token_hidden = pooled_native.unsqueeze(1).expand(
                input_ids.shape[0], input_ids.shape[1], pooled_native.shape[-1]
            )

        pooled = pool(token_hidden, mask_bool, self.config_pooling_mode())
        return BackboneOutput(
            token_hidden=token_hidden,
            pooled=pooled,
            embedding=pooled_native,
            attention_mask=mask_bool,
            num_hidden_layers=self.num_hidden_layers,
        )

    def config_pooling_mode(self) -> str:
        return getattr(self, "_pooling_mode", "mean")

    def set_pooling_mode(self, mode: str) -> None:
        self._pooling_mode = mode

    def _native_embedding(self, raw: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the upstream 768-d embedding, mean-pooled and L2-normalized."""
        for attr in ("last_hidden_state", "pooler_output", "last_hidden_states"):
            value = getattr(raw, attr, None)
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                pooled = pool(value, attention_mask.to(torch.bool), "mean")
                return F.normalize(pooled, p=2.0, dim=-1)
        if hasattr(raw, "get_input_embeddings"):
            # Last resort: embed ids directly. Keeps shape contracts intact for
            # stub-like backbones that return nothing usable.
            embedding_matrix = self.model.get_input_embeddings().weight
            return F.normalize(embedding_matrix[input_ids_check(raw)], p=2.0, dim=-1)
        raise ConfigError("backbone output exposes no usable hidden state")


def input_ids_check(raw: Any) -> torch.Tensor:  # pragma: no cover - defensive helper
    ids = getattr(raw, "input_ids", None)
    if ids is None:
        raise ConfigError("backbone output has no input_ids to fall back on")
    return ids


def _disable_modality_towers(model: nn.Module) -> nn.Module:
    """Drop vision/audio submodules for text-only 270M use.

    The upstream model card documents these as independent components, so removing
    them keeps the 270M text target at ~270M parameters instead of 740M.
    """
    for attr in ("vision_tower", "audio_tower", "vision_model", "audio_model"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except (AttributeError, TypeError):  # pragma: no cover
                pass
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("vision_config", "audio_config"):
            if hasattr(config, attr):
                try:
                    setattr(config, attr, None)
                except (AttributeError, TypeError):  # pragma: no cover
                    pass
    return model


class _TinyTextTower(nn.Module):
    """Minimal deterministic stand-in used by tests.

    Mirrors the real tower's contract (token states + native embedding) at a few
    hundred parameters, so unit tests exercise the real code paths without a
    378 MB download.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        embedding_dim: int = 48,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(512, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        self.config = _TinyConfig(hidden_size, embedding_dim, num_hidden_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **_ignored: Any,
    ) -> "_TinyOutput":
        hidden = self.embed_tokens(input_ids)
        states: List[torch.Tensor] = [hidden]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
            states.append(hidden)
        native = self.projection(hidden)
        return _TinyOutput(
            last_hidden_state=native,
            hidden_states=tuple(states) if output_hidden_states else None,
            attention_mask=attention_mask,
        )


class _TinyConfig:
    def __init__(self, hidden_size: int, embedding_dim: int, num_layers: int) -> None:
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_hidden_layers = num_layers


class _TinyOutput:
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask


class _StubTokenizer:
    """Whitespace/char tokenizer good enough for unit tests.

    Deliberately UTF-8 friendly: it indexes characters, so any language survives.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size

    def __call__(self, texts, max_length=None, truncation=False, return_tensors=None, **_kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        batch = []
        for text in texts:
            ids = [min(ord(ch) % (self.vocab_size - 1), self.vocab_size - 2) for ch in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            ids = ids or [1]
            batch.append(ids)
        if return_tensors != "pt":
            ids = batch[0] if single else batch
            return {"input_ids": ids}
        width = max(len(ids) for ids in batch)
        input_ids = torch.zeros(len(batch), width, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in ids if not skip_special_tokens or token != 0)


# ``BackboneConfig`` has no vocab_size field; the stub needs one number.
BackboneConfig.vocab_size_guess = property(lambda self: 512)


def build_backbone(
    config: Optional[BackboneConfig] = None,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> EmbeddingGemma2TextBackbone:
    return EmbeddingGemma2TextBackbone(config=config, model=model, tokenizer=tokenizer)
