"""Evaluation configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from .base import SubSectionParser


@dataclass
class ClipEvaluationConfig(SubSectionParser):
    """Configuration options for CLIP-based dataset evaluation."""

    SECTION: ClassVar[str] = "clip_evaluation"

    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    precision: str = "fp32"
    normalize_features: bool = True
    output_directory: Path = Path("artifacts/extracted/datasets")
    cache_directory_override: Path | None = None
    batches_per_shard: int = 8_192
    pad_to_full_shard: bool = True
    dataset_names: Sequence[str] | None = None
    limit_batches_per_split: int | None = None
    extra_load_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class TestTransformerConfig(SubSectionParser):
    """Configuration for testing the TransformerTrainer model."""

    SECTION: ClassVar[str] = "test_transformer"

    checkpoint_path: Path
    device: str = "cuda"
    output_directory: Path = Path("artifacts/test_results")
    output_filename: str = "transformer_test_results.csv"