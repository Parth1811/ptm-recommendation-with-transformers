"""Evaluation configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .base import SubSectionParser


@dataclass
class ClipEvaluationConfig(SubSectionParser):
    """Configuration options for CLIP-based ImageNet evaluation."""

    SECTION: ClassVar[str] = "clip_evaluation"

    model_name: str = "openai/clip-vit-base-patch32"
    dataset_split: str = "validation"
    device: str | None = None
    precision: str = "fp32"
    normalize_features: bool = True
    num_batches: int = 1
    output_directory: Path = Path("artifacts/extracted/imagenet_clip")
    cache_directory_override: Path | None = None
