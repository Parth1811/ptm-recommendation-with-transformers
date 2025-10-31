"""Model configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from .base import SubSectionParser


@dataclass
class ModelAutoEncoderConfig(SubSectionParser):
    """Configuration for training the AutoEncoder model."""

    SECTION: ClassVar[str] = "autoencoder"

    encoder_input_size: int
    encoder_output_size: int
    decoder_output_size: int
    encoder_hidden_layers: Sequence[int] = field(default_factory=list)
    decoder_hidden_layers: Sequence[int] = field(default_factory=list)
    use_activation: bool = True

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    shuffle: bool = True

    extracted_models_dir: Path = Path("artifacts/extracted/models")
    model_save_directory: Path = Path("artifacts/models/model_autoencoder")
