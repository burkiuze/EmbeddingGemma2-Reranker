"""Batching collator for reranker training.

Turns :class:`~training.dataset.RerankingExample` objects into padded tensors.
Two things make reranker batching different from ordinary sequence classification:

* **one query, many documents** — the query is encoded once and tiled per
  candidate, so its tokens are not re-sent through the backbone N times;
* **ragged candidate counts** — examples with different numbers of documents are
  padded and masked so the heads never score a padding slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from embeddinggemma_reranker.config import PromptTemplates

from .dataset import RerankingExample


@dataclass
class RerankerBatch:
    """A padded batch of candidate lists."""

    #: ``(B, Tq)`` tokenized queries.
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    #: ``(B, N, Td)`` tokenized documents.
    document_input_ids: torch.Tensor
    document_attention_mask: torch.Tensor
    #: ``(B, N)`` boolean mask marking real (non-padding) candidates.
    document_mask: torch.Tensor
    #: ``(B, N)`` graded relevance.
    relevance: torch.Tensor
    #: ``(B, N)`` teacher scores, or ``None`` when absent from the batch.
    teacher_scores: Optional[torch.Tensor] = None
    #: ``(B, N)`` graded relevance expressed as 0/1 for BCE.
    binary_labels: Optional[torch.Tensor] = None
    #: ``(B, N)`` valid-pair mask (real candidate and supervised).
    pair_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self.query_input_ids.shape[0]

    @property
    def num_candidates(self) -> int:
        return self.document_input_ids.shape[1]

    def to_device(self, device: torch.device) -> "RerankerBatch":
        def move(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.to(device)

        return RerankerBatch(
            query_input_ids=self.query_input_ids.to(device),
            query_attention_mask=self.query_attention_mask.to(device),
            document_input_ids=self.document_input_ids.to(device),
            document_attention_mask=self.document_attention_mask.to(device),
            document_mask=self.document_mask.to(device),
            relevance=self.relevance.to(device),
            teacher_scores=move(self.teacher_scores),
            binary_labels=move(self.binary_labels),
            pair_mask=move(self.pair_mask),
        )

    def flatten_documents(self) -> Dict[str, torch.Tensor]:
        """Collapse ``(B, N, T)`` documents into the ``(B*N, T)`` the encoders want."""
        batch, num, tokens = self.document_input_ids.shape
        return {
            "input_ids": self.document_input_ids.reshape(batch * num, tokens),
            "attention_mask": self.document_attention_mask.reshape(batch * num, tokens),
        }


class RerankerCollator:
    """Tokenize and pad a list of examples into a :class:`RerankerBatch`."""

    def __init__(
        self,
        tokenizer: Any,
        max_query_length: int = 64,
        max_document_length: int = 256,
        prompts: Optional[PromptTemplates] = None,
        binary_threshold: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self.prompts = prompts
        self.binary_threshold = binary_threshold

    def _format_query(self, text: str) -> str:
        if self.prompts is None or not self.prompts.enabled:
            return text
        return self.prompts.apply_query(text)

    def _format_document(self, text: str) -> str:
        if self.prompts is None or not self.prompts.enabled:
            return text
        return self.prompts.apply_document(text)

    def _tokenize(self, texts: Sequence[str], max_length: int) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return dict(batch) if not isinstance(batch, dict) else batch

    def __call__(self, examples: Sequence[RerankingExample]) -> RerankerBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")

        queries = [self._format_query(e.query) for e in examples]
        query_batch = self._tokenize(queries, self.max_query_length)

        max_candidates = max(len(e.documents) for e in examples)
        flat_documents: List[str] = []
        for example in examples:
            flat_documents.extend(self._format_document(d) for d in example.documents)
        document_batch = self._tokenize(flat_documents, self.max_document_length)

        batch_size = len(examples)
        doc_tokens = document_batch["input_ids"].shape[1]

        document_input_ids = torch.zeros(
            batch_size, max_candidates, doc_tokens, dtype=torch.long
        )
        document_attention_mask = torch.zeros(
            batch_size, max_candidates, doc_tokens, dtype=torch.long
        )
        document_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
        relevance = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
        teacher = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
        has_teacher = False

        cursor = 0
        for row, example in enumerate(examples):
            count = len(example.documents)
            document_input_ids[row, :count] = document_batch["input_ids"][
                cursor : cursor + count
            ]
            document_attention_mask[row, :count] = document_batch["attention_mask"][
                cursor : cursor + count
            ]
            document_mask[row, :count] = True
            relevance[row, :count] = torch.tensor(
                example.relevance, dtype=torch.float32
            )
            if example.teacher_scores is not None:
                teacher[row, :count] = torch.tensor(
                    example.teacher_scores, dtype=torch.float32
                )
                has_teacher = True
            cursor += count

        binary_labels = (relevance > self.binary_threshold).to(torch.float32)

        return RerankerBatch(
            query_input_ids=query_batch["input_ids"],
            query_attention_mask=query_batch["attention_mask"],
            document_input_ids=document_input_ids,
            document_attention_mask=document_attention_mask,
            document_mask=document_mask,
            relevance=relevance,
            teacher_scores=teacher if has_teacher else None,
            binary_labels=binary_labels,
            pair_mask=document_mask.clone(),
        )


class PairCollator(RerankerCollator):
    """Collator for strict positive/negative pairs.

    Useful when the loss is purely pairwise: it orders every candidate list as
    ``[positive, negative...]`` and records the index of the positive so the loss
    can index it directly instead of searching for the maximum relevance.
    """

    def __call__(self, examples: Sequence[RerankingExample]) -> RerankerBatch:
        batch = super().__call__(examples)
        positives = torch.tensor(
            [e.best_index for e in examples], dtype=torch.long
        )
        setattr(batch, "positive_indices", positives)
        return batch
