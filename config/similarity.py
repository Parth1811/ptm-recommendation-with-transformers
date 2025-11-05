"""Configuration helpers for model/dataset similarity training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from .base import SubSectionParser
from .train import BaseTrainerConfig


@dataclass
class ModelEmbeddingLoaderConfig(SubSectionParser):
    """Configuration for loading pre-computed model embeddings."""

    SECTION: ClassVar[str] = "model_embedding_loader"

    root_dir: Path = Path("models/embeddings")
    embedding_key: str = "embedding"
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class DatasetTokenLoaderConfig(SubSectionParser):
    """Configuration for loading dataset token shards."""

    SECTION: ClassVar[str] = "dataset_token_loader"

    root_dir: Path = Path("artifacts/extracted/datasets")
    dataset_names: Sequence[str] | None = None
    splits: Sequence[str] = ("train", "validation", "test")
    shard_glob: str = "*.npz"
    batch_size: int = 1
    include_class_metadata: bool = True
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class SimilarityModelConfig(SubSectionParser):
    """Configuration for the similarity transformer model."""

    SECTION: ClassVar[str] = "similarity_model"

    embedding_dim: int = 512
    hidden_dim: int = 768
    num_models: int = 30
    transformer_layers: int = 4
    transformer_heads: int = 8
    intermediate_dim: int | None = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.1
    max_position_embeddings: int = 2048
    use_pretrained: bool = False
    pretrained_model_name: str | None = None


@dataclass
class SimilarityTrainerConfig(BaseTrainerConfig):
    """Training configuration for the similarity transformer."""

    SECTION: ClassVar[str] = "train_similarity_transformer"

    shuffle: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1
    ranking_loss_weight: float = 1.0
    logit_l2_weight: float = 0.0
    extra_loss_weight: float = 0.0
