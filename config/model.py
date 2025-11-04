"""Model architecture configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from .base import SubSectionParser


@dataclass
class ModelAutoEncoderConfig(SubSectionParser):
    """Configuration for the AutoEncoder model architecture."""

    SECTION: ClassVar[str] = "autoencoder"

    encoder_input_size: int
    encoder_output_size: int
    decoder_output_size: int
    encoder_hidden_layers: Sequence[int] = field(default_factory=list)
    decoder_hidden_layers: Sequence[int] = field(default_factory=list)
    use_activation: bool = True
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class ModelAutoEncoderEvalConfig(SubSectionParser):
    """Configuration for evaluating the AutoEncoder on extracted model parameters."""

    SECTION: ClassVar[str] = "autoencoder_evaluation"

    weights_path: Path = Path("artifacts/models/model_autoencoder/latest.pt")
    parameter_root: Path = Path("artifacts/extracted/models")
    output_directory: Path = Path("models/embeddings")
    batch_size: int = 256
    device: str | None = None
    normalize_inputs: bool = True
    flatten: bool = True
    input_dtype: str = "float32"
    file_substring: str = "model_extracted"
    save_dtype: str = "float32"
