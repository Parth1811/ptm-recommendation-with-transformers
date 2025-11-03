"""Dataset configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .base import SubSectionParser


@dataclass
class ImageNetDatasetConfig(SubSectionParser):
    """Configuration for the ImageNet dataset loader."""

    SECTION: ClassVar[str] = "imagenet_dataset"

    dataset_name: str = "ILSVRC/imagenet-1k"
    split: str = "train"
    cache_dir: Path = Path("artifacts/data/imagenet")
    image_size: int = 224
    resize_shorter_side: int = 256
    drop_last: bool = True
    shuffle: bool = True
    seed: int | None = 42
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
