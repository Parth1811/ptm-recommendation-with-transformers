"""PyTorch dataset helpers for the ImageNet-1k benchmark."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from config import ConfigParser, ImageNetDatasetConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_transform(split: str, image_size: int, resize_shorter_side: int) -> transforms.Compose:
    """Return the canonical torchvision transform pipeline for ImageNet."""
    if split.lower() == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    # Validation / test style preprocessing
    return transforms.Compose(
        [
            transforms.Resize(resize_shorter_side),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageNetDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset wrapper around the HuggingFace ImageNet-1k release."""

    def __init__(
        self,
        config: ImageNetDatasetConfig | None = None,
        *,
        split: str | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        ConfigParser.load()
        cfg = config or ConfigParser.get(ImageNetDatasetConfig)
        split_name = split or cfg.split
        if cache_dir is not None:
            cache_root = Path(cache_dir).expanduser()
        else:
            cache_root = cfg.cache_dir.expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)

        self.hf_dataset = load_dataset(
            cfg.dataset_name,
            split=split_name,
            cache_dir=str(cache_root),
        )

        if "image" not in self.hf_dataset.column_names:
            raise KeyError("Expected 'image' column in ImageNet dataset.")
        if "label" not in self.hf_dataset.column_names:
            raise KeyError("Expected 'label' column in ImageNet dataset.")

        self.transform = transform or _default_transform(split_name, cfg.image_size, cfg.resize_shorter_side)
        self.split = split_name
        self.cache_dir = cache_root

        self.labels: list[int] = [int(label) for label in self.hf_dataset["label"]]
        if not self.labels:
            raise ValueError("Loaded ImageNet dataset is empty.")

        feature_labels = self.hf_dataset.features.get("label")
        self.label_names: Sequence[str] = (
            feature_labels.names if feature_labels is not None and hasattr(feature_labels, "names") else tuple()
        )
        self.num_classes = len(self.label_names) if self.label_names else len(set(self.labels))

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record: dict[str, Any] = self.hf_dataset[int(index)]
        image = record["image"]

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(np.array(image)).convert("RGB")

        tensor = self.transform(pil_image) if self.transform else transforms.ToTensor()(pil_image)
        label = int(record["label"])
        return tensor, label


class ClassBalancedBatchSampler(Sampler[list[int]]):
    """Sampler that yields batches containing one sample for each class."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if not labels:
            raise ValueError("Cannot build a sampler without labels.")

        self.labels = np.asarray(labels, dtype=np.int64)
        self.unique_classes = np.unique(self.labels)
        if len(self.unique_classes) == 0:
            raise ValueError("No unique classes found in labels.")

        self.label_to_indices: dict[int, np.ndarray] = {
            int(label): np.flatnonzero(self.labels == label) for label in self.unique_classes
        }
        if any(len(indices) == 0 for indices in self.label_to_indices.values()):
            raise ValueError("Each class must contain at least one sample for balanced batching.")

        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        counts = [len(indices) for indices in self.label_to_indices.values()]
        if self.drop_last:
            return min(counts)
        return max(counts)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(None if self.seed is None else self.seed + self._epoch)
        self._epoch += 1

        ordered_indices: dict[int, np.ndarray] = {}
        for label, indices in self.label_to_indices.items():
            ordered_indices[label] = rng.permutation(indices) if self.shuffle else indices.copy()

        if self.drop_last:
            num_batches = min(len(indices) for indices in ordered_indices.values())
        else:
            num_batches = max(len(indices) for indices in ordered_indices.values())

        for batch_idx in range(num_batches):
            batch: list[int] = []
            for label in self.unique_classes:
                class_indices = ordered_indices[int(label)]
                if batch_idx >= len(class_indices):
                    if self.drop_last:
                        return
                    idx = class_indices[-1]
                else:
                    idx = class_indices[batch_idx]
                batch.append(int(idx))
            yield batch

    def set_epoch(self, epoch: int) -> None:
        """Allow external schedulers to control shuffling deterministically."""
        self._epoch = epoch


class ImageNetBalancedDataLoader(DataLoader[tuple[torch.Tensor, int]]):
    """DataLoader that returns ImageNet batches with one sample per class."""

    def __init__(
        self,
        dataset: ImageNetDataset | None = None,
        *,
        config: ImageNetDatasetConfig | None = None,
        split: str | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        ConfigParser.load()
        cfg = config or ConfigParser.get(ImageNetDatasetConfig)

        if dataset is None:
            dataset = ImageNetDataset(
                config=cfg,
                split=split,
                transform=transform,
                cache_dir=cache_dir,
            )

        sampler = ClassBalancedBatchSampler(
            dataset.labels,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
        }

        if cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = cfg.persistent_workers
            if cfg.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

        # Remove conflicting arguments if provided
        kwargs.pop("batch_size", None)
        kwargs.pop("shuffle", None)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        loader_kwargs.update(kwargs)

        super().__init__(**loader_kwargs)

        self.dataset: ImageNetDataset = dataset
        self.balanced_sampler = sampler
        self.config = cfg

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def class_names(self) -> Sequence[str]:
        return self.dataset.label_names
