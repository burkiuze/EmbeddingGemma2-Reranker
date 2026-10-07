"""Training loop and parameter-freezing policy.

The four modes differ only in which modules get gradients:

===============  ===========================================  ==================
mode             trainable                                    typical use
===============  ===========================================  ==================
``head_only``    relevance + confidence heads                  cheapest probe
``reranker_only`` fusion + interaction + reranker layers + heads  **default**
``lora``         the above plus LoRA adapters on the backbone   memory-aware
``full``         everything                                    explicit opt-in only
===============  ===========================================  ==================

``full`` is never selected implicitly and the loop refuses to start it without an
explicit ``output_dir`` plus an ``allow_full_finetune`` acknowledgement.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from embeddinggemma_reranker.config import RerankerConfig
from embeddinggemma_reranker.model import EmbeddingGemma2Reranker, _repeat_backbone
from embeddinggemma_reranker.statistics import count_parameters, count_trainable

from .collator import RerankerBatch, RerankerCollator
from .dataset import RerankingExample
from .losses import LossOutput, compute_loss


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible runs."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU here
        torch.cuda.manual_seed_all(seed)


class FullFinetuneNotConfirmed(RuntimeError):
    """Raised when ``full`` mode is requested without explicit acknowledgement."""


@dataclass
class TrainableGroups:
    """Which parameter groups exist and their learning rates."""

    head: List[nn.Parameter] = field(default_factory=list)
    body: List[nn.Parameter] = field(default_factory=list)
    backbone: List[nn.Parameter] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(g) for g in (self.head, self.body, self.backbone))

    def parameter_count(self) -> int:
        return sum(p.numel() for g in (self.head, self.body, self.backbone) for p in g)


def configure_trainable(
    model: EmbeddingGemma2Reranker, mode: str
) -> TrainableGroups:
    """Freeze/unfreeze modules per ``mode`` and collect the parameter groups."""
    if mode not in ("head_only", "reranker_only", "lora", "full"):
        raise ValueError(
            f"unknown train_mode {mode!r}; expected head_only, reranker_only, lora or full"
        )

    def set_requires(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    if mode == "full":
        set_requires(model.backbone, True)
    else:
        set_requires(model.backbone, False)

    if mode == "head_only":
        set_requires(model.fusion, False)
        set_requires(model.reranker, False)
        set_requires(model.token_interaction, False)
        set_requires(model.relevance_head, True)
        set_requires(model.confidence_head, True)
    else:
        set_requires(model.fusion, True)
        set_requires(model.reranker, True)
        set_requires(model.token_interaction, True)
        set_requires(model.relevance_head, True)
        set_requires(model.confidence_head, True)

    if mode == "lora":
        _attach_lora(model)

    groups = TrainableGroups()
    groups.head = [p for p in model.relevance_head.parameters() if p.requires_grad]
    if model.confidence_head is not None:
        groups.head += [p for p in model.confidence_head.parameters() if p.requires_grad]

    for module in (model.fusion, model.reranker, model.token_interaction):
        if module is not None:
            groups.body += [p for p in module.parameters() if p.requires_grad]

    groups.backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    return groups


def _attach_lora(model: EmbeddingGemma2Reranker) -> None:
    """Wrap eligible backbone projections with PEFT LoRA when available."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        # PEFT is optional; without it `lora` degrades to `reranker_only` and we
        # say so rather than pretending adapters were added.
        model._lora_degraded = True
        return

    config = model.config.training
    targets = list(
        config.lora_target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    peft_model = get_peft_model(model.backbone.model, lora_config)
    model.backbone.model = peft_model
    for param in peft_model.parameters():
        param.requires_grad = "lora" in param.name


@dataclass
class TrainingSummary:
    """Everything a run produced, written to ``train_log.jsonl`` / metadata."""

    train_mode: str
    trainable_parameters: int
    total_parameters: int
    trainable_percentage: float
    epochs: int
    steps: int
    steps_completed: int
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    output_dir: str = ""
    loss_name: str = ""
    lora_degraded: bool = False
    wall_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_mode": self.train_mode,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_percentage": self.trainable_percentage,
            "epochs": self.epochs,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "best_loss": self.best_loss,
            "final_loss": self.final_loss,
            "output_dir": self.output_dir,
            "loss_name": self.loss_name,
            "lora_degraded": self.lora_degraded,
            "wall_seconds": round(self.wall_seconds, 3),
            "metrics": self.metrics,
        }


class RerankerTrainer:
    """Minimal, dependency-light trainer for the reranker."""

    def __init__(
        self,
        model: EmbeddingGemma2Reranker,
        config: RerankerConfig,
        collator: Optional[RerankerCollator] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.training = config.training
        self.device = torch.device(
            self.training.output_dir and config.backbone.device or config.backbone.device
        )
        self.collator = collator or RerankerCollator(
            tokenizer=model.backbone.tokenizer,
            max_query_length=self.training.max_query_length,
            max_document_length=self.training.max_document_length,
            prompts=config.prompts,
        )
        self.history: List[Dict[str, Any]] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler = None

    # ------------------------------------------------------------ parameter groups
    def prepare(self, allow_full_finetune: bool = False) -> TrainableGroups:
        """Freeze according to the mode and build the optimizer."""
        if self.training.train_mode == "full" and not allow_full_finetune:
            raise FullFinetuneNotConfirmed(
                "train_mode='full' fine-tunes the entire 270M backbone. Pass "
                "allow_full_finetune=True (CLI: --allow-full-finetune) to confirm."
            )

        groups = configure_trainable(self.model, self.training.train_mode)
        if groups.total == 0:
            raise ValueError(
                f"train_mode={self.training.train_mode!r} left no trainable parameters"
            )

        param_groups: List[Dict[str, Any]] = []
        head_lr = self.training.head_learning_rate or self.training.learning_rate
        if groups.head:
            param_groups.append({"params": groups.head, "lr": head_lr, "name": "head"})
        if groups.body:
            param_groups.append({"params": groups.body, "lr": self.training.learning_rate, "name": "body"})
        if groups.backbone:
            param_groups.append({"params": groups.backbone, "lr": self.training.learning_rate, "name": "backbone"})

        self._optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.training.weight_decay
        )
        return groups

    def _make_scheduler(self, total_steps: int):
        if total_steps <= 0:
            return None
        warmup = int(total_steps * self.training.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda)

    # ------------------------------------------------------------------ scoring
    def score_batch(self, batch: RerankerBatch) -> torch.Tensor:
        """Run the model over a batch, returning ``(B, N)`` raw scores.

        The query is encoded once per row and tiled across candidates, and padded
        candidate slots are masked to a neutral value so they cannot influence
        listwise softmax or the confidence head.
        """
        query_output = self.model.encode_query_batch(
            batch.query_input_ids, batch.query_attention_mask
        )
        documents = batch.flatten_documents()
        document_output = self.model.backbone(**documents)

        batch_size, num_candidates = batch.document_mask.shape
        pair_queries = _repeat_backbone(query_output, batch_size * num_candidates)
        output = self.model.score_pairs(pair_queries, document_output)

        scores = output.scores.reshape(batch_size, num_candidates)
        neutral = scores.detach().mean()
        return torch.where(batch.document_mask, scores, neutral)

    def loss_for(
        self, batch: RerankerBatch, scores: torch.Tensor
    ) -> LossOutput:
        return compute_loss(
            self.training.loss,
            scores,
            batch,
            margin=self.training.margin,
            listwise_temperature=self.training.listwise_temperature,
            distillation_weight=self.training.distillation_weight,
        )

    # -------------------------------------------------------------------- loop
    def train(
        self,
        examples: Sequence[RerankingExample],
        allow_full_finetune: bool = False,
        log_every: Optional[int] = None,
    ) -> TrainingSummary:
        """Run the configured number of epochs and return a summary."""
        if not examples:
            raise ValueError("cannot train on an empty dataset")

        started = time.perf_counter()
        set_seed(self.training.seed)

        groups = self.prepare(allow_full_finetune=allow_full_finetune)
        loader = DataLoader(
            list(examples),
            batch_size=self.training.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            drop_last=False,
        )

        total_steps = max(1, len(loader) * self.training.epochs)
        self._scheduler = self._make_scheduler(total_steps)

        self.model.train()
        if self.training.train_mode != "full":
            self.model.backbone.eval()

        step = 0
        best_loss: Optional[float] = None
        final_loss: Optional[float] = None
        last_metrics: Dict[str, float] = {}
        report_every = log_every if log_every is not None else self.training.log_every

        for epoch in range(self.training.epochs):
            for index, batch in enumerate(loader):
                batch = batch.to_device(self.model.device)
                scores = self.score_batch(batch)
                output = self.loss_for(batch, scores)

                if not torch.isfinite(output.loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch} step {index}: "
                        f"{float(output.loss)}"
                    )

                loss = output.loss / self.training.gradient_accumulation_steps
                loss.backward()

                if (index + 1) % self.training.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    step += 1
                    if self._scheduler is not None:
                        self._scheduler.step()

                value = float(output.loss.item())
                final_loss = value
                best_loss = value if best_loss is None else min(best_loss, value)
                last_metrics = dict(output.metrics)

                if report_every and (index + 1) % report_every == 0:
                    record = {
                        "epoch": epoch,
                        "step": step,
                        "loss": value,
                        "learning_rate": (
                            self._optimizer.param_groups[0]["lr"]
                            if self._optimizer
                            else None
                        ),
                        **last_metrics,
                    }
                    self.history.append(record)
                    print(
                        f"epoch {epoch} step {index + 1}/{len(loader)} "
                        f"loss={value:.4f} trainable={groups.parameter_count():,}",
                        flush=True,
                    )

        summary = TrainingSummary(
            train_mode=self.training.train_mode,
            trainable_parameters=count_trainable(self.model),
            total_parameters=count_parameters(self.model),
            trainable_percentage=round(
                100.0 * count_trainable(self.model) / max(1, count_parameters(self.model)), 4
            ),
            epochs=self.training.epochs,
            steps=total_steps,
            steps_completed=step,
            best_loss=best_loss,
            final_loss=final_loss,
            output_dir=self.training.output_dir,
            loss_name=self.training.loss,
            lora_degraded=bool(getattr(self.model, "_lora_degraded", False)),
            wall_seconds=time.perf_counter() - started,
            metrics=last_metrics,
        )
        self.model.eval()
        return summary

    def _optimizer_step(self) -> None:
        if self._optimizer is None:  # pragma: no cover - prepare() always runs
            raise RuntimeError("optimizer not initialised; call prepare() first")
        if self.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self._optimizer.param_groups for p in group["params"]],
                self.training.max_grad_norm,
            )
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ saving
    def save(self, summary: TrainingSummary) -> Path:
        output_dir = Path(self.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.model.state_dict(), output_dir / "reranker_state.pt"
        )
        self.model.config.save(output_dir / "reranker_config.json")
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "train_log.jsonl").write_text(
            "\n".join(json.dumps(record) for record in self.history) + "\n",
            encoding="utf-8",
        )
        return output_dir


def print_trainable_summary(model: EmbeddingGemma2Reranker, mode: str) -> None:
    """Print the actual trainable/total counts. Never estimated."""
    trainable = count_trainable(model)
    total = count_parameters(model)
    percentage = 100.0 * trainable / max(1, total)
    print(
        f"train_mode={mode}\n"
        f"  trainable parameters : {trainable:,}\n"
        f"  frozen parameters    : {total - trainable:,}\n"
        f"  total parameters     : {total:,}\n"
        f"  trainable percentage : {percentage:.2f}%"
    )
