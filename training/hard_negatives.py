"""Hard-negative mining.

Optional and not required for basic training. When enabled, the *original*
EmbeddingGemma bi-encoder is used as a cheap first-stage retriever: embed the
corpus, take the top-k most similar documents per query, drop the known positives,
and keep what is left as hard negatives.

    Embedding search -> top-k similar documents -> remove positives
                    -> hard negatives -> reranker training

The reranker being trained is never used as its own retriever, so mining cannot
collapse onto the model's current biases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from embeddinggemma_reranker.config import RerankerConfig
from embeddinggemma_reranker.model import EmbeddingGemma2Reranker

from .dataset import RerankingExample, save_jsonl

NEGATIVE_RANDOM = "random"
NEGATIVE_HARD = "hard"
NEGATIVE_POSITIVE = "positive"


@dataclass
class MiningConfig:
    """Knobs for the mining pass."""

    top_k: int = 10
    num_hard_negatives: int = 2
    batch_size: int = 8
    similarity: str = "cosine"
    remove_positives: bool = True
    #: Textual near-duplicate guard; cheap and deliberately conservative.
    min_text_overlap: float = 0.0

    def validate(self) -> "MiningConfig":
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.num_hard_negatives <= 0:
            raise ValueError(
                f"num_hard_negatives must be positive, got {self.num_hard_negatives}"
            )
        if self.similarity not in ("cosine", "dot"):
            raise ValueError(
                f"similarity must be 'cosine' or 'dot', got {self.similarity!r}"
            )
        if not 0.0 <= self.min_text_overlap <= 1.0:
            raise ValueError("min_text_overlap must be in [0, 1]")
        return self


@dataclass
class MiningStats:
    """What mining actually did, for the run log."""

    num_examples: int = 0
    num_corpus: int = 0
    num_positives_removed: int = 0
    num_hard_negatives_added: int = 0
    num_random_negatives_added: int = 0
    examples_without_negatives: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_corpus": self.num_corpus,
            "num_positives_removed": self.num_positives_removed,
            "num_hard_negatives_added": self.num_hard_negatives_added,
            "num_random_negatives_added": self.num_random_negatives_added,
            "examples_without_negatives": self.examples_without_negatives,
            "fallback_reasons": self.fallback_reasons,
        }


def _text_key(text: str) -> str:
    return " ".join(text.lower().split())


def similarity_matrix(
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    method: str = "cosine",
) -> torch.Tensor:
    """Score matrix ``(Q, D)`` for the retrieval baseline."""
    if query_embeddings.dim() != 2 or document_embeddings.dim() != 2:
        raise ValueError("embeddings must be 2-D (N, D)")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError(
            f"embedding widths differ: {query_embeddings.shape[1]} vs "
            f"{document_embeddings.shape[1]}"
        )
    if method == "cosine":
        query_embeddings = F.normalize(query_embeddings, p=2.0, dim=-1)
        document_embeddings = F.normalize(document_embeddings, p=2.0, dim=-1)
    return query_embeddings @ document_embeddings.T


@torch.no_grad()
def embed_corpus(
    model: EmbeddingGemma2Reranker, documents: Sequence[str]
) -> torch.Tensor:
    """Embed a corpus with the backbone's native 768-d output."""
    outputs: List[torch.Tensor] = []
    for start in range(0, len(documents), 8):
        chunk = list(documents[start : start + 8])
        encoded = model.document_encoder(chunk)
        if encoded.embedding is None:  # pragma: no cover - defensive
            raise ValueError("backbone did not expose a native embedding")
        outputs.append(encoded.embedding)
    if not outputs:
        return torch.zeros(0, model.native_embedding_dim)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def mine_hard_negatives(
    model: EmbeddingGemma2Reranker,
    examples: Sequence[RerankingExample],
    corpus: Sequence[str],
    config: Optional[MiningConfig] = None,
    random_seed: int = 42,
) -> Tuple[List[RerankingExample], MiningStats]:
    """Augment each example with mined hard negatives.

    Returns new examples and a :class:`MiningStats` describing exactly what
    happened, including examples where no negative could be found.
    """
    config = (config or MiningConfig()).validate()
    stats = MiningStats(num_examples=len(examples), num_corpus=len(corpus))

    corpus_keys = [_text_key(doc) for doc in corpus]
    corpus_index: Dict[str, int] = {}
    for position, key in enumerate(corpus_keys):
        corpus_index.setdefault(key, position)

    corpus_embeddings = embed_corpus(model, corpus)

    import random as _random

    rng = _random.Random(random_seed)
    augmented: List[RerankingExample] = []

    for example in examples:
        positives = {positives_key(example, idx) for idx in example.positives}
        if config.remove_positives:
            stats.num_positives_removed += len(positives)

        candidate_slots = [
            position
            for position, key in enumerate(corpus_keys)
            if key not in positives
        ]

        ranked: List[int] = []
        if corpus_embeddings.numel() and example.query:
            query_output = model.encode_query(example.query)
            if query_output.embedding is not None:
                scores = similarity_matrix(
                    query_output.embedding.unsqueeze(0),
                    corpus_embeddings,
                    config.similarity,
                )[0]
                order = torch.argsort(scores, descending=True)[: config.top_k]
                ranked = [
                    slot
                    for slot in order.tolist()
                    if corpus_keys[slot] not in positives
                ]

        hard_texts: List[str] = []
        for slot in ranked:
            if len(hard_texts) >= config.num_hard_negatives:
                break
            text = corpus[slot]
            if config.min_text_overlap > 0:
                if _overlap(example.query, text) >= config.min_text_overlap:
                    stats.fallback_reasons["duplicate_positive"] = (
                        stats.fallback_reasons.get("duplicate_positive", 0) + 1
                    )
                    continue
            if text not in hard_texts:
                hard_texts.append(text)

        random_texts: List[str] = []
        attempts = 0
        target_random = max(0, config.num_hard_negatives - len(hard_texts))
        while len(random_texts) < target_random and attempts < 50:
            attempts += 1
            if not candidate_slots:
                break
            slot = candidate_slots[rng.randrange(len(candidate_slots))]
            text = corpus[slot]
            if text not in hard_texts and text not in random_texts:
                random_texts.append(text)

        negatives = hard_texts + random_texts
        if not negatives:
            stats.examples_without_negatives += 1
            stats.fallback_reasons["empty_corpus_or_all_positive"] = (
                stats.fallback_reasons.get("empty_corpus_or_all_positive", 0) + 1
            )

        documents = list(example.documents)
        relevance = list(example.relevance)
        kinds = list(example.negative_types or ["positive"] * len(example.documents))

        for text in hard_texts:
            documents.append(text)
            relevance.append(0.0)
            kinds.append(NEGATIVE_HARD)
        for text in random_texts:
            documents.append(text)
            relevance.append(0.0)
            kinds.append(NEGATIVE_RANDOM)

        stats.num_hard_negatives_added += len(hard_texts)
        stats.num_random_negatives_added += len(random_texts)

        augmented.append(
            RerankingExample(
                query=example.query,
                documents=documents,
                relevance=relevance,
                teacher_scores=(
                    None
                    if example.teacher_scores is None
                    else example.teacher_scores + [0.0] * len(negatives)
                ),
                negative_types=kinds,
                metadata={**example.metadata, "mined": True},
            )
        )

    return augmented, stats


def positives_key(example: RerankingExample, index: int) -> str:
    return _text_key(example.documents[index])


def _overlap(left: str, right: str) -> float:
    """Jaccard overlap of whitespace tokens; used only as a duplicate guard."""
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_corpus(path: str | Path, text_key: str = "text") -> List[str]:
    """Load a retrieval corpus from JSONL.

    Accepts ``{"text": ...}``, ``{"content": ...}``, ``{"body": ...}`` or a bare
    JSON string per line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")

    documents: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if isinstance(record, str):
                documents.append(record)
            elif isinstance(record, dict):
                for key in (text_key, "text", "content", "body", "passage"):
                    if key in record and isinstance(record[key], str):
                        documents.append(record[key])
                        break
                else:
                    raise ValueError(
                        f"{path}:{line_number}: no text field found in {sorted(record)}"
                    )
            else:
                raise ValueError(f"{path}:{line_number}: unsupported record type")
    return documents


def write_mined(
    path: str | Path,
    examples: Sequence[RerankingExample],
    stats: MiningStats,
) -> None:
    """Persist mined examples plus a sibling ``*.stats.json`` audit file."""
    save_jsonl(path, examples)
    stats_path = Path(str(path) + ".stats.json")
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")


def mining_from_config(config: RerankerConfig) -> MiningConfig:
    return MiningConfig(
        top_k=max(4, config.training.num_negatives * 4),
        num_hard_negatives=config.training.num_negatives,
    )
