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
    max_models: int | None = None
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class DatasetTokenLoaderConfig(SubSectionParser):
    """Configuration for loading dataset token shards."""

    SECTION: ClassVar[str] = "dataset_token_loader"

    root_dir: Path = Path("artifacts/extracted/datasets")
    dataset_names: Sequence[str] | None = None
    shard_glob: str = "*.npz"
    batch_size: int = 1
    shuffle: bool = True
    include_class_metadata: bool = True
    max_shards: int | None = None
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
    temperature_init: float = 0.07
    temperature_min: float = 0.01
    temperature_max: float = 5.0


@dataclass
class SimilarityTrainerConfig(BaseTrainerConfig):
    """Training configuration for the similarity transformer."""

    SECTION: ClassVar[str] = "train_similarity_transformer"

    shuffle: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_min_lr: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    gradient_clip_norm: float = 1.0
    num_epochs: int = 1
    train_splits: Sequence[str] = ("train",)
    validation_splits: Sequence[str] = ("validation",)
    validation_interval_epochs: int = 1
    ranking_loss_weight: float = 0.1
    logit_l2_weight: float = 0.0
    extra_loss_weight: float = 0.0
    label_smoothing: float = 0.1
    temperature_init: float = 0.07
    temperature_min: float = 0.01
    temperature_max: float = 5.0
    hard_negative_top_k: int = 4
    hard_negative_margin: float = 0.1
    hard_negative_weight: float = 1.0
    overfit_subset_size: int | None = 64
    overfit_max_epochs: int = 50
    overfit_shuffle: bool = True
    enable_overfit_check: bool = True
    log_grad_stats: bool = True
    log_activation_stats: bool = True
    model_save_directory: Path = Path("artifacts/models/similarity_transformer")


@dataclass
class CustomSimilarityTransformerConfig(SubSectionParser):
    """Configuration for the custom cross-attention similarity transformer."""

    SECTION: ClassVar[str] = "custom_similarity_transformer"

    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    batch_first: bool = True
