"""The reranker model: shared backbone -> encoders -> fusion -> layers -> heads.

This is the primary architecture. The bi-encoder cosine baseline lives in
``evaluation/baseline.py`` and is deliberately *not* what
:class:`EmbeddingGemma2Reranker` does: the backbone's 768-d embedding is read for
comparison and hard-negative mining, but the score itself comes from a learned
head over fused query/document features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import BackboneOutput, EmbeddingGemma2TextBackbone
from .config import BackboneConfig, RerankerConfig
from .fusion import QueryDocumentFusion
from .heads import ConfidenceHead, RelevanceHead
from .interaction import TokenInteraction
from .reranker_layers import RerankerStack


@dataclass
class RerankerOutput:
    """Everything one forward pass produces."""

    #: ``(P,)`` raw relevance logits, one per (query, document) pair.
    scores: torch.Tensor
    #: ``(P, D)`` ranking representations the head consumed.
    representations: torch.Tensor
    #: ``(P,)`` optional confidence logits; ``None`` when the head is disabled.
    confidence: Optional[torch.Tensor] = None
    #: ``(P,)`` cosine similarity of the native embeddings, for comparison only.
    native_similarity: Optional[torch.Tensor] = None
    #: Per-pair intermediates, available when ``return_details=True``.
    details: Dict[str, torch.Tensor] = field(default_factory=dict)


class EncoderBase(nn.Module):
    """Shared plumbing for query and document encoding.

    Both sides call the *same* :class:`EmbeddingGemma2TextBackbone` instance by
    default, so a 270M tower is resident in memory once rather than twice.
    """

    #: Prefix template applied by :meth:`format_inputs`.
    prefix: str = ""

    def __init__(
        self,
        backbone: EmbeddingGemma2TextBackbone,
        prompts: Optional[Any] = None,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompts = prompts
        self.max_length = max_length
        #: Set by the model so a query encoded once can be reused per candidate.
        self._cache: Dict[Any, BackboneOutput] = {}

    def format_inputs(
        self, texts: Sequence[str], titles: Optional[Sequence[Optional[str]]] = None
    ) -> List[str]:
        """Apply the asymmetric prompt convention from the upstream config."""
        if self.prompts is None or not getattr(self.prompts, "enabled", True):
            return list(texts)
        if self.prefix == "query":
            return [self.prompts.apply_query(t) for t in texts]
        if titles is not None:
            return [
                self.prompts.apply_document(t, title)
                for t, title in zip(texts, titles)
            ]
        return [self.prompts.apply_document(t) for t in texts]

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_key(self, text: str) -> Any:
        return (self.prefix, text, self.max_length)

    def encode(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        use_cache: bool = False,
    ) -> BackboneOutput:
        if not texts:
            raise ValueError("cannot encode an empty list of texts")

        if use_cache:
            missing = [t for t in texts if self._cache_key(t) not in self._cache]
            if missing:
                fresh = self._encode_uncached(missing, titles)
                for text, out in zip(missing, fresh):
                    self._cache[self._cache_key(text)] = out
            outputs = [self._cache[self._cache_key(t)] for t in texts]
            return _stack_outputs(outputs)

        return self._encode_uncached(list(texts), titles)

    def _encode_uncached(
        self,
        texts: List[str],
        titles: Optional[Sequence[Optional[str]]] = None,
    ) -> BackboneOutput:
        formatted = self.format_inputs(texts, titles)
        encoded = self.tokenize(formatted)
        return self.backbone(**encoded)

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        batch = self.backbone.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if isinstance(batch, dict):
            return batch
        return dict(batch)


class QueryEncoder(EncoderBase):
    """Encodes the query side with a task prefix."""

    prefix = "query"


class DocumentEncoder(EncoderBase):
    """Encodes candidate documents, optionally with titles."""

    prefix = "document"


def _stack_outputs(outputs: List[BackboneOutput]) -> BackboneOutput:
    if len(outputs) == 1:
        return outputs[0]
    return BackboneOutput(
        token_hidden=torch.cat([o.token_hidden for o in outputs], dim=0),
        pooled=torch.cat([o.pooled for o in outputs], dim=0),
        embedding=None
        if outputs[0].embedding is None
        else torch.cat([o.embedding for o in outputs], dim=0),
        attention_mask=torch.cat([o.attention_mask for o in outputs], dim=0),
        num_hidden_layers=outputs[0].num_hidden_layers,
    )


class EmbeddingGemma2Reranker(nn.Module):
    """Cross-encoder / interaction-based reranker on the EmbeddingGemma 2 text tower.

    Pipeline::

        query   -> shared backbone -> query repr
        doc     -> shared backbone -> doc repr
                                 -> fusion -> [optional token interaction]
                                 -> reranker layers -> relevance head -> score
    """

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        backbone: Optional[EmbeddingGemma2TextBackbone] = None,
    ) -> None:
        super().__init__()
        self.config = (config or RerankerConfig()).validate()

        self.backbone = backbone or EmbeddingGemma2TextBackbone(self.config.backbone)
        if not self.config.interaction.enabled and self.config.fusion.interaction_type == "token":
            self.config.interaction.enabled = True

        self.query_encoder = QueryEncoder(
            self.backbone, self.config.prompts, self.config.training.max_query_length
        )
        self.document_encoder = DocumentEncoder(
            self.backbone, self.config.prompts, self.config.training.max_document_length
        )

        self.backbone_hidden = self.backbone.hidden_size
        self.native_embedding_dim = self.backbone.embedding_dim

        self.fusion = QueryDocumentFusion.from_config(
            query_dim=self.backbone_hidden,
            document_dim=self.backbone_hidden,
            config=self.config.fusion,
        )

        self.token_interaction: Optional[TokenInteraction] = None
        extra_dims: List[int] = []
        if self.config.interaction.enabled:
            self.token_interaction = TokenInteraction.from_config(
                self.backbone_hidden, self.config.interaction
            )
            extra_dims.append(self.config.interaction.projection_dim)

        self.reranker = RerankerStack.from_config(
            input_dim=self.config.fusion.fusion_dim,
            config=self.config.reranker_layers,
            extra_features=extra_dims,
        )

        total_extra = sum(extra_dims)
        self.relevance_head = RelevanceHead.from_config(
            input_dim=self.reranker.output_dim,
            config=self.config.heads,
            extra_features=0,
        )
        self.confidence_head: Optional[ConfidenceHead] = None
        if self.config.heads.confidence_head:
            self.confidence_head = ConfidenceHead.from_config(
                self.reranker.output_dim, self.config.heads
            )
        self._total_extra = total_extra

    # ---------------------------------------------------------- properties
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    @property
    def shares_backbone(self) -> bool:
        """Query and document encoders use the same weights by default."""
        return self.query_encoder.backbone is self.document_encoder.backbone

    # -------------------------------------------------------------- encode
    def encode_query(self, query: str, use_cache: bool = True) -> BackboneOutput:
        return self.query_encoder([query], use_cache=use_cache)

    def encode_query_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """Encode an already-tokenized query batch.

        The collator tokenizes queries, so training feeds ids straight through
        rather than re-tokenizing text and risking a train/serve mismatch.
        """
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def encode_documents(
        self,
        documents: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
    ) -> BackboneOutput:
        return self.document_encoder(documents, titles=titles)

    def clear_cache(self) -> None:
        self.query_encoder.clear_cache()
        self.document_encoder.clear_cache()

    # ------------------------------------------------------------- scoring
    def score_pairs(
        self,
        query_output: BackboneOutput,
        document_output: BackboneOutput,
        return_details: bool = False,
    ) -> RerankerOutput:
        """Score ``P`` pre-aligned (query, document) pairs.

        ``query_output.pooled[i]`` is paired with ``document_output.pooled[i]``.
        """
        if query_output.pooled.shape[0] != document_output.pooled.shape[0]:
            raise ValueError(
                f"pair count mismatch: {query_output.pooled.shape[0]} queries vs "
                f"{document_output.pooled.shape[0]} documents"
            )

        fused = self.fusion(query_output.pooled, document_output.pooled)
        extras: List[torch.Tensor] = []

        if self.token_interaction is not None:
            pair_count = query_output.pooled.shape[0]
            if query_output.token_hidden.shape[0] != pair_count:
                raise ValueError(
                    "token interaction needs per-pair token states; "
                    "encode with output_hidden=True"
                )
            token_out = self.token_interaction(
                query_output.token_hidden,
                document_output.token_hidden,
                query_mask=query_output.attention_mask,
                document_mask=document_output.attention_mask,
            )
            extras.append(token_out["token_context"])
            if return_details:
                fused["interaction_matrix"] = token_out["interaction_matrix"]

        representations = self.reranker(fused["fused"], extras or None)
        scores = self.relevance_head(representations)

        native_similarity = None
        if (
            query_output.embedding is not None
            and document_output.embedding is not None
        ):
            native_similarity = torch.einsum(
                "pd,pd->p",
                query_output.embedding,
                document_output.embedding,
            )

        details: Dict[str, torch.Tensor] = {}
        if return_details:
            details["fused"] = fused["fused"]
            details["query_projected"] = fused["query_projected"]
            details["document_projected"] = fused["document_projected"]

        return RerankerOutput(
            scores=scores,
            representations=representations,
            confidence=None,
            native_similarity=native_similarity,
            details=details,
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        return_details: bool = False,
    ) -> RerankerOutput:
        """Score one query against N documents.

        The query is encoded **once** and reused for every candidate, which is the
        main reason this is faster than re-encoding per pair.
        """
        if not documents:
            raise ValueError("cannot rerank an empty candidate list")

        query_output = self.encode_query(query)
        # Repeat the query representation once per candidate so the head sees
        # aligned pairs.
        pair_queries = _repeat_backbone(query_output, len(documents))
        document_output = self.encode_documents(documents, titles=titles)
        return self.score_pairs(
            pair_queries, document_output, return_details=return_details
        )

    def confidence(
        self, representations: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Confidence logits for one query's candidate set.

        ``representations`` is ``(N, D)``; it is treated as a single candidate list.
        """
        if self.confidence_head is None:
            from .heads import confidence_from_scores

            return confidence_from_scores(
                scores.reshape(1, -1), num_documents=scores.shape[0]
            ).squeeze(0)
        batched = representations.unsqueeze(0)
        return self.confidence_head(batched, scores.reshape(1, -1)).squeeze(0)


def _repeat_backbone(output: BackboneOutput, times: int) -> BackboneOutput:
    """Tile a single encoded sequence ``times`` times to form aligned pairs."""
    return BackboneOutput(
        token_hidden=output.token_hidden.expand(times, *output.token_hidden.shape[1:]),
        pooled=output.pooled.expand(times, -1),
        embedding=None
        if output.embedding is None
        else output.embedding.expand(times, -1),
        attention_mask=output.attention_mask.expand(times, *output.attention_mask.shape[1:]),
        num_hidden_layers=output.num_hidden_layers,
    )


def build_reranker(
    config: Optional[RerankerConfig] = None,
    backbone: Optional[EmbeddingGemma2TextBackbone] = None,
) -> EmbeddingGemma2Reranker:
    return EmbeddingGemma2Reranker(config=config, backbone=backbone)


def build_from_backbone_config(
    backbone_config: BackboneConfig,
    **kwargs: Any,
) -> EmbeddingGemma2Reranker:
    config = RerankerConfig(backbone=backbone_config)
    for key, value in kwargs.items():
        setattr(config, key, value)
    return EmbeddingGemma2Reranker(config=config.validate())
