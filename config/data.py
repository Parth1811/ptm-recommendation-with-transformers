"""Dataset configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from .base import SubSectionParser


@dataclass
class DatasetLoaderDefaultsConfig(SubSectionParser):
    """Shared defaults applied to dataset loaders."""

    SECTION: ClassVar[str] = "dataset_defaults"

    cache_dir: Path = Path("artifacts/data")
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int | None = 42
    drop_last: bool = False
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    image_column: str | None = "image"
    label_column: str | None = "label"
    balance_classes: bool = True
    max_samples_train: int | None = None
    max_samples_validation: int | None = None
    max_samples_test: int | None = None


@dataclass
class DatasetRegistryConfig(SubSectionParser):
    """Registry mapping dataset names to loader implementations."""

    SECTION: ClassVar[str] = "dataset_loader"

    default_loader_class: str = "GenericBalancedDataLoader"
    loader_registry: dict[str, dict[str, object]] = field(default_factory=dict)
