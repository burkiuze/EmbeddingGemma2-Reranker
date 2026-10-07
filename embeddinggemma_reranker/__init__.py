"""EmbeddingGemma2-Reranker.

A neural cross-encoder built on top of Google's EmbeddingGemma 2 270M text
tower. The backbone is used unchanged; every reranking-specific module is new and
lives above it.

    EmbeddingGemma 2 text backbone
      + query/document representations
      + query-document fusion
      + optional token interaction
      + trainable reranker layers
      + relevance head
      + optional confidence / abstention
      + ranking API
"""

from .backbone import BackboneOutput, EmbeddingGemma2TextBackbone, build_backbone
from .config import (
    BackboneConfig,
    ChunkingConfig,
    ConfigError,
    FusionConfig,
    HeadConfig,
    InteractionConfig,
    PolicyConfig,
    PromptTemplates,
    RerankerConfig,
    RerankerLayerConfig,
    TrainingConfig,
    apply_cli_overrides,
    config_summary,
)
from .fusion import QueryDocumentFusion, elementwise_features
from .heads import (
    ConfidenceHead,
    RelevanceHead,
    calibrate,
    confidence_from_scores,
)
from .inference import (
    EmbeddingGemma2RerankerInference,
    RerankResponse,
    RerankedResult,
    aggregate_chunk_scores,
    chunk_document,
)
from .interaction import LateInteractionScorer, TokenInteraction, build_interaction
from .model import (
    DocumentEncoder,
    EmbeddingGemma2Reranker,
    QueryEncoder,
    RerankerOutput,
    build_reranker,
)
from .policy import RankingDecision, RankingPolicy, build_ranking, sort_scores
from .pooling import pool
from .reranker_layers import RerankerLayer, RerankerStack
from .statistics import (
    ParameterReport,
    breakdown,
    build_report,
    count_frozen,
    count_parameters,
    count_trainable,
    estimate_memory,
    format_report,
    freeze,
)

__version__ = "0.1.0"

__all__ = [
    "BackboneConfig",
    "BackboneOutput",
    "ChunkingConfig",
    "ConfigError",
    "ConfidenceHead",
    "DocumentEncoder",
    "EmbeddingGemma2Reranker",
    "EmbeddingGemma2RerankerInference",
    "EmbeddingGemma2TextBackbone",
    "FusionConfig",
    "HeadConfig",
    "InteractionConfig",
    "LateInteractionScorer",
    "ParameterReport",
    "PolicyConfig",
    "PromptTemplates",
    "QueryDocumentFusion",
    "QueryEncoder",
    "RerankResponse",
    "RerankerConfig",
    "RerankerLayer",
    "RerankerLayerConfig",
    "RerankerOutput",
    "RerankerStack",
    "RerankedResult",
    "RelevanceHead",
    "RankingDecision",
    "RankingPolicy",
    "TokenInteraction",
    "TrainingConfig",
    "aggregate_chunk_scores",
    "apply_cli_overrides",
    "breakdown",
    "build_backbone",
    "build_interaction",
    "build_ranking",
    "build_report",
    "build_reranker",
    "calibrate",
    "chunk_document",
    "confidence_from_scores",
    "config_summary",
    "count_frozen",
    "count_parameters",
    "count_trainable",
    "elementwise_features",
    "estimate_memory",
    "format_report",
    "freeze",
    "pool",
    "sort_scores",
    "__version__",
]
