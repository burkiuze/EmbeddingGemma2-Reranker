"""User-facing inference: chunking, scoring, ranking, confidence, abstention.

This is the layer that turns a raw model into the API described in the project
brief::

    reranker.rerank(query="...", documents=[...])

Long documents are chunked and aggregated rather than silently truncated, and any
truncation that did happen is reported in the result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .config import RerankerConfig
from .heads import calibrate
from .model import EmbeddingGemma2Reranker
from .policy import RankingPolicy, build_ranking


@dataclass
class RerankedResult:
    """One candidate's final output."""

    index: int
    rank: int
    score: float
    document: Optional[str] = None
    document_id: Optional[Any] = None
    calibrated_score: Optional[float] = None
    confidence: Optional[float] = None
    chunked: bool = False
    num_chunks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": self.index,
            "rank": self.rank,
            "score": self.score,
        }
        if self.calibrated_score is not None:
            out["calibrated_score"] = self.calibrated_score
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.document_id is not None:
            out["document_id"] = self.document_id
        if self.document is not None:
            out["document"] = self.document
        if self.chunked:
            out["chunked"] = True
            out["num_chunks"] = self.num_chunks
        return out


@dataclass
class RerankResponse:
    """Full response for one query."""

    query: str
    results: List[RerankedResult] = field(default_factory=list)
    ranking: List[int] = field(default_factory=list)
    insufficient_confidence: bool = False
    abstention_reason: Optional[str] = None
    top_score: Optional[float] = None
    margin: Optional[float] = None
    confidence: Optional[float] = None
    truncated_documents: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": self.query,
            "ranking": self.ranking,
            "results": [r.to_dict() for r in self.results],
            "insufficient_confidence": self.insufficient_confidence,
        }
        if self.abstention_reason:
            payload["abstention_reason"] = self.abstention_reason
        if self.top_score is not None:
            payload["top_score"] = self.top_score
        if self.margin is not None:
            payload["margin"] = self.margin
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.truncated_documents:
            payload["truncated_documents"] = self.truncated_documents
        if self.metrics:
            payload["metrics"] = self.metrics
        return payload


def chunk_document(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    max_chunks: int = 8,
    stride: int = 0,
) -> List[str]:
    """Split ``text`` into at most ``max_chunks`` chunks of ``max_tokens`` tokens.

    Returns the chunk strings re-decoded from token ids, so chunks are exactly
    what the model will see. Short documents come back as a single-element list.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")

    ids = tokenizer(text, add_special_tokens=False)
    input_ids = _extract_ids(ids)

    if len(input_ids) <= max_tokens:
        return [text]

    step = max_tokens - max(0, stride)
    if step <= 0:
        step = max_tokens

    chunks: List[int] = []
    for start in range(0, len(input_ids), step):
        piece = input_ids[start : start + max_tokens]
        if not piece:
            break
        chunks.append(piece)
        if len(chunks) >= max_chunks:
            break

    decoded: List[str] = []
    for piece in chunks:
        try:
            decoded.append(tokenizer.decode(piece, skip_special_tokens=True))
        except TypeError:  # pragma: no cover - tokenizers without decode
            decoded.append(text)
    return decoded or [text]


def _extract_ids(tokenized: Any) -> List[int]:
    if isinstance(tokenized, dict):
        for key in ("input_ids", "ids"):
            if key in tokenized:
                value = tokenized[key]
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if value and isinstance(value[0], list):
                    value = value[0]
                return list(value)
        return []
    if hasattr(tokenized, "input_ids"):
        value = tokenized.input_ids
        return value.tolist() if hasattr(value, "tolist") else list(value)
    if isinstance(tokenized, list):
        return tokenized
    return []


def aggregate_chunk_scores(
    chunk_scores: Sequence[float], aggregation: str = "max", top_k: int = 3
) -> float:
    """Combine per-chunk scores into one document score."""
    if not chunk_scores:
        raise ValueError("cannot aggregate an empty chunk score list")

    values = torch.tensor(list(chunk_scores), dtype=torch.float32)
    if aggregation == "max":
        return float(values.max().item())
    if aggregation == "mean":
        return float(values.mean().item())
    if aggregation == "top_k_mean":
        k = max(1, min(top_k, values.numel()))
        top = torch.topk(values, k).values
        return float(top.mean().item())
    raise ValueError(
        f"unknown chunk aggregation {aggregation!r}; expected max, mean or top_k_mean"
    )


class EmbeddingGemma2RerankerInference:
    """High-level API around :class:`EmbeddingGemma2Reranker`."""

    def __init__(
        self,
        model: EmbeddingGemma2Reranker,
        config: Optional[RerankerConfig] = None,
        policy: Optional[RankingPolicy] = None,
    ) -> None:
        self.model = model
        self.config = config or model.config
        self.policy = policy or RankingPolicy(self.config.policy)
        self.model.eval()

    # ------------------------------------------------------------- chunking
    def _prepare_documents(
        self, documents: Sequence[str]
    ) -> Dict[int, List[str]]:
        """Return ``{document_index: chunks}``, flagging anything that was split."""
        prepared: Dict[int, List[str]] = {}
        if not self.config.chunking.enabled:
            for index, document in enumerate(documents):
                prepared[index] = [document]
            return prepared

        for index, document in enumerate(documents):
            prepared[index] = chunk_document(
                document,
                self.model.backbone.tokenizer,
                max_tokens=self.config.chunking.max_tokens,
                max_chunks=self.config.chunking.max_chunks,
                stride=self.config.chunking.stride,
            )
        return prepared

    # -------------------------------------------------------------- scoring
    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        document_ids: Optional[Sequence[Any]] = None,
        top_k: Optional[int] = None,
        calibrate_scores: bool = True,
        include_confidence: bool = True,
    ) -> RerankResponse:
        """Rank ``documents`` against ``query``.

        Parameters
        ----------
        query: the search query.
        documents: candidate texts.
        document_ids: optional identifiers echoed back on each result.
        top_k: keep only the first k results (ranking is still computed over all).
        calibrate_scores: attach a sigmoid probability alongside the raw score.
        include_confidence: run the confidence head when it is enabled.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        documents = list(documents)
        if not documents:
            raise ValueError("cannot rerank an empty candidate list")
        if document_ids is not None and len(document_ids) != len(documents):
            raise ValueError(
                f"document_ids length {len(document_ids)} does not match "
                f"{len(documents)} documents"
            )

        started = time.perf_counter()
        self.model.clear_cache()

        chunks = self._prepare_documents(documents)
        flattened: List[str] = []
        owner: List[int] = []
        for doc_index, chunk_list in chunks.items():
            for chunk in chunk_list:
                flattened.append(chunk)
                owner.append(doc_index)

        output = self.model.score(query, flattened)

        # Aggregate chunk scores back up to document level.
        per_document: Dict[int, List[float]] = {}
        for doc_index, score in zip(owner, output.scores.tolist()):
            per_document.setdefault(doc_index, []).append(score)

        document_scores = [
            aggregate_chunk_scores(
                per_document[index],
                self.config.chunking.aggregation,
                self.config.chunking.top_k,
            )
            for index in range(len(documents))
        ]
        score_tensor = torch.tensor(document_scores, dtype=torch.float32)

        ranking = build_ranking(
            score_tensor, document_ids if document_ids is not None else list(range(len(documents)))
        )

        # Confidence operates on the document-level ranking representations. Pick
        # the representation belonging to each document's best chunk.
        best_slot = _best_chunk_per_document(owner, output.scores)
        document_representations = output.representations[best_slot].unsqueeze(0)

        confidence_value: Optional[float] = None
        if include_confidence and self.model.confidence_head is not None:
            logits = self.model.confidence(
                output.representations[best_slot], document_scores_tensor(score_tensor)
            )
            confidence_value = float(torch.sigmoid(logits).item())

        sorted_scores = torch.sort(score_tensor, descending=True).values
        top_score = float(sorted_scores[0].item())
        margin = (
            float((sorted_scores[0] - sorted_scores[1]).item())
            if sorted_scores.numel() > 1
            else None
        )

        decision = self.policy.evaluate(
            top_score=torch.tensor(top_score),
            margin=None if margin is None else torch.tensor(margin),
            confidence=None if confidence_value is None else torch.tensor(confidence_value),
        )

        results: List[RerankedResult] = []
        for position, item in enumerate(ranking["ranking"]):
            index = item["index"]
            # ``index`` is a document_id when supplied, so map back by position.
            position_in_input = (
                documents.index(documents[0]) if False else _position_of(index, document_ids, documents)
            )
            chunk_list = chunks[position_in_input]
            results.append(
                RerankedResult(
                    index=index,
                    rank=position + 1,
                    score=item["score"],
                    document=documents[position_in_input],
                    document_id=document_ids[position_in_input]
                    if document_ids is not None
                    else None,
                    calibrated_score=float(
                        calibrate(torch.tensor(item["score"]), self.config.calibration).item()
                    )
                    if calibrate_scores
                    else None,
                    chunked=len(chunk_list) > 1,
                    num_chunks=len(chunk_list),
                )
            )

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            results = results[:top_k]

        elapsed = time.perf_counter() - started
        truncated = [i for i, c in chunks.items() if len(c) > 1]

        return RerankResponse(
            query=query,
            results=results,
            ranking=[r.index for r in results],
            insufficient_confidence=not decision.accepted,
            abstention_reason=decision.reason,
            top_score=top_score,
            margin=margin,
            confidence=confidence_value,
            truncated_documents=truncated,
            metrics={
                "num_documents": len(documents),
                "num_scored_segments": len(flattened),
                "latency_seconds": round(elapsed, 6),
                "documents_per_second": (
                    round(len(documents) / elapsed, 3) if elapsed > 0 else None
                ),
                "aggregation": self.config.chunking.aggregation,
            },
        )

    def rerank_batch(
        self,
        queries: Sequence[str],
        document_lists: Sequence[Sequence[str]],
        **kwargs: Any,
    ) -> List[RerankResponse]:
        """Rerank several queries against their own candidate lists."""
        if len(queries) != len(document_lists):
            raise ValueError(
                f"queries ({len(queries)}) and document lists ({len(document_lists)}) "
                "must have the same length"
            )
        return [
            self.rerank(query, documents, **kwargs)
            for query, documents in zip(queries, document_lists)
        ]


def document_scores_tensor(scores: torch.Tensor) -> torch.Tensor:
    """Reshape a document-level score vector into the head's ``(1, N)`` layout."""
    return scores.reshape(1, -1)


def _best_chunk_per_document(owner: Sequence[int], scores: torch.Tensor) -> torch.Tensor:
    """Indices of the highest-scoring chunk for each document."""
    values = scores.tolist()
    best: Dict[int, int] = {}
    for slot, doc_index in enumerate(owner):
        current = best.get(doc_index)
        if current is None or values[slot] > values[current]:
            best[doc_index] = slot
    return torch.tensor([best[i] for i in sorted(best)], dtype=torch.long)


def _position_of(
    index: int, document_ids: Optional[Sequence[Any]], documents: Sequence[str]
) -> int:
    """Map a result index back to its position in the input list."""
    if document_ids is None:
        return index
    try:
        return list(document_ids).index(index)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"document_id {index!r} not found in the input list") from exc
