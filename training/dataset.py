"""Modular JSONL dataset loaders.

Three input shapes are supported, all plain UTF-8 JSONL so any source (BEIR,
MS MARCO, a hand-written file, a teacher log) can be adapted with a one-line
change of the key names:

Pairwise::

    {"query": "...", "positive": "...", "negative": "..."}

Multi-candidate with graded relevance::

    {"query": "...", "documents": ["...", "..."], "relevance": [0, 2, 1]}

Teacher scores for distillation::

    {"query": "...", "documents": ["...", "..."], "teacher_scores": [8.2, 1.3]}

No dataset is committed to this repository; these loaders read whatever you point
them at.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from embeddinggemma_reranker.config import ConfigError


@dataclass
class RerankingExample:
    """One query with its candidate documents and supervision."""

    query: str
    documents: List[str]
    #: Relevance grades; higher is better. Derived from positions when absent.
    relevance: List[float] = field(default_factory=list)
    #: Externally supplied teacher scores. Never fabricated.
    teacher_scores: Optional[List[float]] = None
    #: Free-form provenance so hard negatives stay distinguishable from random ones.
    negative_types: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ValueError(f"query must be a string, got {type(self.query).__name__}")
        if not self.documents:
            raise ValueError("example must contain at least one document")
        if not all(isinstance(d, str) for d in self.documents):
            raise ValueError("all documents must be strings")
        if self.relevance and len(self.relevance) != len(self.documents):
            raise ValueError(
                f"relevance length {len(self.relevance)} does not match "
                f"{len(self.documents)} documents"
            )
        if self.teacher_scores is not None and len(self.teacher_scores) != len(
            self.documents
        ):
            raise ValueError(
                f"teacher_scores length {len(self.teacher_scores)} does not match "
                f"{len(self.documents)} documents"
            )
        if not self.relevance:
            # Degenerate case: everything is a positive. Document it rather than
            # inventing a ranking.
            self.relevance = [1.0] * len(self.documents)

    @property
    def positives(self) -> List[int]:
        return [i for i, r in enumerate(self.relevance) if r > 0]

    @property
    def negatives(self) -> List[int]:
        return [i for i, r in enumerate(self.relevance) if r <= 0]

    @property
    def best_index(self) -> int:
        return max(range(len(self.relevance)), key=lambda i: self.relevance[i])

    def subset(self, indices: Sequence[int]) -> "RerankingExample":
        return RerankingExample(
            query=self.query,
            documents=[self.documents[i] for i in indices],
            relevance=[self.relevance[i] for i in indices],
            teacher_scores=None
            if self.teacher_scores is None
            else [self.teacher_scores[i] for i in indices],
            negative_types=None
            if self.negative_types is None
            else [self.negative_types[i] for i in indices],
            metadata=dict(self.metadata),
        )


def _as_list(value: Any, field_name: str) -> List[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
    out: List[float] = []
    for item in value:
        if isinstance(item, bool):
            out.append(float(item))
        elif isinstance(item, (int, float)):
            out.append(float(item))
        else:
            raise ValueError(f"{field_name} must contain numbers, got {item!r}")
    return out


def parse_pairwise(record: Dict[str, Any], num_negatives: int = 1) -> List[RerankingExample]:
    """Expand one pairwise row into candidate-list examples.

    With ``num_negatives=1`` this yields a single 2-document example. Larger values
    emit multiple examples, each pairing the positive with a different negative —
    the standard way to get more gradient signal from pairwise data.
    """
    query = record.get("query")
    positive = record.get("positive")
    if not isinstance(query, str) or not isinstance(positive, str):
        raise ValueError("pairwise records need string 'query' and 'positive'")

    negatives = record.get("negatives")
    if negatives is None:
        single = record.get("negative")
        negatives = [single] if isinstance(single, str) else []
    if not isinstance(negatives, list):
        raise ValueError("'negatives' must be a list when present")
    negatives = [n for n in negatives if isinstance(n, str)]
    if not negatives:
        raise ValueError(
            "pairwise record has no negative; supply 'negative' or 'negatives'"
        )

    kinds = record.get("negative_types")
    examples: List[RerankingExample] = []
    for group_start in range(0, len(negatives), num_negatives):
        chosen = negatives[group_start : group_start + num_negatives]
        if not chosen:
            break
        documents = [positive, *chosen]
        relevance = [1.0, *([0.0] * len(chosen))]
        types = ["positive", *(kinds[group_start : group_start + len(chosen)] if kinds else ["negative"] * len(chosen))]
        examples.append(
            RerankingExample(
                query=query,
                documents=documents,
                relevance=relevance,
                negative_types=types,
                metadata={"source": "pairwise"},
            )
        )
    return examples


def parse_candidates(record: Dict[str, Any]) -> RerankingExample:
    """Parse a multi-candidate row, accepting several key spellings."""
    query = record.get("query")
    documents = record.get("documents")
    if not isinstance(query, str):
        raise ValueError("'query' must be a string")
    if not isinstance(documents, list) or not documents:
        raise ValueError("'documents' must be a non-empty list")

    relevance_raw = record.get("relevance")
    if relevance_raw is None:
        relevance_raw = record.get("scores")
    if relevance_raw is None:
        relevance = [1.0] * len(documents)
    else:
        relevance = _as_list(relevance_raw, "relevance")

    if len(relevance) != len(documents):
        raise ValueError(
            f"relevance has {len(relevance)} entries but there are "
            f"{len(documents)} documents"
        )

    teacher = record.get("teacher_scores")
    if teacher is not None:
        teacher = _as_list(teacher, "teacher_scores")

    kinds = record.get("negative_types")
    return RerankingExample(
        query=query,
        documents=list(documents),
        relevance=relevance,
        teacher_scores=teacher,
        negative_types=list(kinds) if kinds else None,
        metadata={k: v for k, v in record.items() if k not in ("query", "documents", "relevance", "scores", "teacher_scores")},
    )


def parse_record(record: Dict[str, Any], num_negatives: int = 1) -> List[RerankingExample]:
    """Dispatch one JSON object to the right parser."""
    if "documents" in record:
        return [parse_candidates(record)]
    if "positive" in record or "negative" in record or "negatives" in record:
        return parse_pairwise(record, num_negatives=num_negatives)
    raise ValueError(
        "record has neither 'documents' nor 'positive'/'negative'; "
        f"keys were {sorted(record)}"
    )


def load_jsonl(
    path: str | Path,
    num_negatives: int = 1,
    limit: Optional[int] = None,
) -> List[RerankingExample]:
    """Read a JSONL file into examples.

    Blank lines and ``#`` comments are skipped. Malformed rows raise with the
    1-based line number rather than being dropped silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    examples: List[RerankingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be an object")
            try:
                examples.extend(parse_record(record, num_negatives=num_negatives))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if limit is not None and len(examples) >= limit:
                return examples[:limit]
    return examples


def iter_jsonl(path: str | Path, num_negatives: int = 1) -> Iterator[RerankingExample]:
    """Streaming variant of :func:`load_jsonl` for large corpora."""
    yield from load_jsonl(path, num_negatives=num_negatives)


def split_examples(
    examples: Sequence[RerankingExample],
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[List[RerankingExample], List[RerankingExample]]:
    """Deterministic train/validation split."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ConfigError(
            f"validation_ratio must be in [0, 1), got {validation_ratio}"
        )
    if not examples:
        return [], []

    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * (1.0 - validation_ratio))
    train = [examples[i] for i in indices[:cut]]
    validation = [examples[i] for i in indices[cut:]]
    return train, validation


def save_jsonl(path: str | Path, examples: Sequence[RerankingExample]) -> None:
    """Write examples back out in the multi-candidate format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record: Dict[str, Any] = {
                "query": example.query,
                "documents": example.documents,
                "relevance": example.relevance,
            }
            if example.teacher_scores is not None:
                record["teacher_scores"] = example.teacher_scores
            if example.negative_types is not None:
                record["negative_types"] = example.negative_types
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sample_data(directory: str | Path, num_examples: int = 8) -> Dict[str, Path]:
    """Create a tiny sample dataset so the pipeline is runnable out of the box.

    These are synthetic placeholder strings, not benchmark data. They exist to
    exercise the code path, and any metric computed on them is meaningless.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("How do I reset a password?",
         "Password reset instructions: open the account settings page.",
         "How to prepare pasta: boil water and salt it first."),
        ("What is Python?",
         "Python is a high-level programming language.",
         "The Pacific Ocean is the largest ocean on Earth."),
        ("JWT expiration validation",
         "def validate_jwt(token): return not token.is_expired()",
         "def bake_cake(): preheat oven to 180 degrees."),
        ("Deploy to production",
         "Deployment runbook: build, test, then promote the artifact.",
         "A cat sleeps roughly twelve to sixteen hours a day."),
        ("¿Cómo configuro la autenticación?",
         "La autenticación se configura en el archivo de configuración.",
         "El gato negro duerme en el sofá."),
        ("数据库连接如何配置",
         "数据库连接字符串在环境变量中配置。",
         "今天的天气很好，适合散步。"),
        ("How do I configure authentication?",
         "Authentication configuration lives in auth.yaml.",
         "Coffee beans are roasted before brewing."),
        ("What does the reranker score mean?",
         "The relevance head emits an unbounded raw relevance logit.",
         "The cat sat on the mat quietly."),
    ]

    train_path = directory / "train.jsonl"
    eval_path = directory / "eval.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for query, positive, negative in pairs[:max(1, num_examples - 2)]:
            handle.write(
                json.dumps(
                    {"query": query, "positive": positive, "negative": negative},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with eval_path.open("w", encoding="utf-8") as handle:
        for query, positive, negative in pairs[-2:]:
            handle.write(
                json.dumps(
                    {
                        "query": query,
                        "documents": [positive, negative, positive],
                        "relevance": [2.0, 0.0, 1.0],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"train": train_path, "eval": eval_path}
