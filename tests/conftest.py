"""Shared test fixtures.

Every fixture here uses the tiny stub backbone. Unit tests must never download
model weights: a full run of this suite needs no network and no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embeddinggemma_reranker.config import (  # noqa: E402
    BackboneConfig,
    ChunkingConfig,
    FusionConfig,
    HeadConfig,
    InteractionConfig,
    PolicyConfig,
    PromptTemplates,
    RerankerConfig,
    RerankerLayerConfig,
    TrainingConfig,
)

TINY_HIDDEN = 32
TINY_EMBED = 48
TINY_LAYERS = 2


def tiny_backbone_config() -> BackboneConfig:
    """A stub backbone config. Real values are in docs/source_model_analysis.md."""
    return BackboneConfig(
        model_name_or_path="stub://tiny",
        hidden_size=TINY_HIDDEN,
        embedding_dim=TINY_EMBED,
        num_hidden_layers=TINY_LAYERS,
        dtype="float32",
    )


def tiny_config(**overrides: Any) -> RerankerConfig:
    """A small but complete reranker config suitable for unit tests."""
    config = RerankerConfig(
        backbone=tiny_backbone_config(),
        fusion=FusionConfig(fusion_dim=24, dropout=0.0),
        interaction=InteractionConfig(enabled=False, projection_dim=16),
        reranker_layers=RerankerLayerConfig(
            hidden_dim=24, num_layers=2, intermediate_dim=32, dropout=0.0
        ),
        heads=HeadConfig(
            relevance_hidden_dim=16, confidence_hidden_dim=16, relevance_dropout=0.0,
            confidence_dropout=0.0,
        ),
        chunking=ChunkingConfig(enabled=False),
        policy=PolicyConfig(),
        training=TrainingConfig(
            max_query_length=24,
            max_document_length=32,
            batch_size=2,
            output_dir="runs/test",
        ),
        prompts=PromptTemplates(enabled=False),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config.validate()


@pytest.fixture
def backbone_config() -> BackboneConfig:
    return tiny_backbone_config()


@pytest.fixture
def config() -> RerankerConfig:
    return tiny_config()


@pytest.fixture
def model(config: RerankerConfig):
    from embeddinggemma_reranker.model import build_reranker

    torch.manual_seed(0)
    return build_reranker(config)


@pytest.fixture
def interaction_model():
    """A reranker with the optional token-interaction path enabled."""
    torch.manual_seed(0)
    config = tiny_config()
    config.fusion.interaction_type = "token"
    config.interaction.enabled = True
    config.validate()
    from embeddinggemma_reranker.model import build_reranker

    return build_reranker(config)


@pytest.fixture
def examples() -> List[Any]:
    """Deterministic examples with a mix of graded relevance."""
    from training.dataset import RerankingExample

    return [
        RerankingExample(
            query="How do I reset a password?",
            documents=[
                "Password reset instructions: open account settings.",
                "How to prepare pasta: boil water and salt it.",
                "Account security and login help.",
            ],
            relevance=[3.0, 0.0, 1.0],
        ),
        RerankingExample(
            query="¿Cómo configuro la autenticación?",
            documents=[
                "La autenticación se configura en auth.yaml.",
                "El gato duerme en el sofá.",
            ],
            relevance=[2.0, 0.0],
        ),
        RerankingExample(
            query="数据库连接如何配置",
            documents=[
                "数据库连接字符串在环境变量中配置。",
                "今天的天气很好。",
                "配置文件位于 etc 目录。",
            ],
            relevance=[2.0, 0.0, 1.0],
        ),
    ]


@pytest.fixture
def batch(model, examples):
    from training.collator import RerankerCollator

    collator = RerankerCollator(
        tokenizer=model.backbone.tokenizer,
        max_query_length=model.config.training.max_query_length,
        max_document_length=model.config.training.max_document_length,
        prompts=model.config.prompts,
    )
    return collator(examples)
