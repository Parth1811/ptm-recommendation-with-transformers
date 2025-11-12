"""Training configuration helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .base import SubSectionParser


@dataclass
class BaseTrainerConfig(SubSectionParser, ABC):
    """Configuration for training a model."""

    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 32
    device: str = "cuda"

    learning_rate: float = 1e-3
    scheduler_mode: str = "min"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_threshold: float = 1e-4
    scheduler_threshold_mode: str = "rel"
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 1e-5
    scheduler_monitor: str = "validation.loss"
    weight_decay: float = 0.0

    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    log_every_n_epochs: int = 1
    log_every_n_steps: int = 1
    save_checkpoint_every_n_epochs: int = 1
    progress_description: str = "Training"
    model_save_directory: Path = Path("artifacts/models")


@dataclass
class TrainModelAutoEncoderConfig(BaseTrainerConfig):
    """Configuration for training the AutoEncoder model."""

    SECTION: ClassVar[str] = "train_model_autoencoder"
    shuffle: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    normalize_inputs: bool = True
    code_l1_penalty: float = 0.0
    reconstruction_loss: str = "smooth_l1"
    smooth_l1_beta: float = 0.5
    validate_every_n_epochs: int = 1
    extracted_models_dir: Path = Path("artifacts/extracted/models")
    model_save_directory: Path = Path("artifacts/models/model_autoencoder")


@dataclass
class CustomSimilarityTransformerTrainerConfig(BaseTrainerConfig):
    """Training configuration for the CustomSimilarityTransformer model."""

    SECTION: ClassVar[str] = "custom_similarity_transformer_trainer"

    shuffle: bool = True
    regularization_weight: float = 0.1
    validation_interval_epochs: int = 1
    model_save_directory: Path = Path("artifacts/models/custom_similarity_transformer")


@dataclass
class TransformerTrainerConfig(BaseTrainerConfig):
    """Configuration for TransformerTrainer using torch.nn.Transformer."""

    SECTION: ClassVar[str] = "transformer_trainer"

    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_models: int = 16

    # Training hyperparameters
    shuffle: bool = True
    validate_every_n_epochs: int = 1
    ranking_loss_weight: float = 1.0
    smooth_l1_weight: float = 0.1
    model_save_directory: Path = Path("artifacts/models/transformer")
