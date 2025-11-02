"""Training configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .base import SubSectionParser


@dataclass
class TrainModelAutoEncoderConfig(SubSectionParser):
    """Configuration for training the AutoEncoder model."""

    SECTION: ClassVar[str] = "train_model_autoencoder"

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    shuffle: bool = True
    log_every_n_epochs: int = 1
    progress_description: str = "ModelAutoEncoder"

    extracted_models_dir: Path = Path("artifacts/extracted/models")
    model_save_directory: Path = Path("artifacts/models/model_autoencoder")
