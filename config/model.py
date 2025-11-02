"""Model architecture configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
