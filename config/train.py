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
    learning_rate: float = 1e-3
    log_every_n_epochs: int = 1
    progress_description: str = "Training"
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0


@dataclass
class TrainModelAutoEncoderConfig(BaseTrainerConfig):
    """Configuration for training the AutoEncoder model."""

    SECTION: ClassVar[str] = "train_model_autoencoder"
    shuffle: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15
    scheduler_min_lr: float = 1e-5
    scheduler_cooldown: int = 0
    scheduler_threshold: float = 1e-4
    scheduler_threshold_mode: str = "rel"
    scheduler_mode: str = "min"
    normalize_inputs: bool = True
    code_l1_penalty: float = 0.0
    reconstruction_loss: str = "smooth_l1"
    smooth_l1_beta: float = 0.5
    extracted_models_dir: Path = Path("artifacts/extracted/models")
    model_save_directory: Path = Path("artifacts/models/model_autoencoder")
