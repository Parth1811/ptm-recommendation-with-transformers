"""Generic dataset helpers for Hugging Face vision datasets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Callable, Mapping

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets.features import ClassLabel, Image as HfImage
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
from torchvision import transforms


def _detect_image_column(dataset: HFDataset) -> str:
    for column, feature in dataset.features.items():
        if isinstance(feature, HfImage):
            return column
    raise KeyError("Unable to detect an image column; please specify 'image_column' explicitly.")


def _detect_label_column(dataset: HFDataset) -> str:
    candidate: str | None = None
    for column, feature in dataset.features.items():
        if isinstance(feature, ClassLabel):
            return column
        feature_dtype = getattr(feature, "dtype", None)
        if feature_dtype in {"int64", "int32", "int16", "int8"} and candidate is None:
            candidate = column
    if candidate is not None:
        return candidate
    if "label" in dataset.column_names:
        return "label"
    raise KeyError("Unable to detect a label column; please specify 'label_column' explicitly.")


def _prepare_label_encoder(
    dataset: HFDataset,
    label_column: str,
) -> tuple[list[int], Sequence[str], Mapping[str, int] | None]:
    raw_labels = dataset[label_column]
    feature = dataset.features.get(label_column)

    if isinstance(feature, ClassLabel):
        labels = [int(label) for label in raw_labels]
        names: Sequence[str] = tuple(feature.names)
        return labels, names, None

    labels: list[int] = []
    label_names: list[str] = []
    encoder: dict[str, int] | None = None

    try:
        labels = [int(label) for label in raw_labels]
        unique_labels = sorted(set(labels))
        label_names = [str(value) for value in unique_labels]
        return labels, tuple(label_names), None
    except (TypeError, ValueError):
        str_labels = [str(label) for label in raw_labels]
        unique_str = sorted(set(str_labels))
        encoder = {value: idx for idx, value in enumerate(unique_str)}
        labels = [encoder[value] for value in str_labels]
        return labels, tuple(unique_str), encoder


class GenericImageDataset(TorchDataset[tuple[torch.Tensor, int]]):
    """Torch dataset wrapper around a Hugging Face vision dataset split."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_column: str | None = None,
        label_column: str | None = None,
    ) -> None:
        if not isinstance(hf_dataset, HFDataset):
            raise TypeError("Expected a Hugging Face Dataset instance.")

        self.hf_dataset = hf_dataset
        self.image_column = image_column or _detect_image_column(hf_dataset)
        self.label_column = label_column or _detect_label_column(hf_dataset)

        if self.image_column not in hf_dataset.column_names:
            raise KeyError(f"Column '{self.image_column}' not present in dataset.")
        if self.label_column not in hf_dataset.column_names:
            raise KeyError(f"Column '{self.label_column}' not present in dataset.")

        self.transform = transform or transforms.ToTensor()
        self.labels, label_names, encoder = _prepare_label_encoder(hf_dataset, self.label_column)
        if not self.labels:
            raise ValueError("Loaded dataset split is empty.")

        self.label_names: Sequence[str] = label_names
        self.num_classes = len(self.label_names) if self.label_names else len(set(self.labels))
        self._label_encoder = encoder

    @property
    def class_names(self) -> Sequence[str]:
        """Expose human-readable class names when available."""
        return self.label_names

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _encode_label(self, value: Any) -> int:
        if self._label_encoder is None:
            return int(value)
        return self._label_encoder[str(value)]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record: Mapping[str, Any] = self.hf_dataset[int(index)]
        image_value = record[self.image_column]

        if isinstance(image_value, Image.Image):
            pil_image = image_value.convert("RGB")
        else:
            pil_image = Image.fromarray(np.array(image_value)).convert("RGB")

        tensor = self.transform(pil_image) if self.transform else transforms.ToTensor()(pil_image)
        label_value = record[self.label_column]
        label = self._encode_label(label_value)
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


class GenericBalancedDataLoader(DataLoader[tuple[torch.Tensor, int]]):
    """DataLoader that returns balanced batches with one sample per class."""

    def __init__(
        self,
        dataset: GenericImageDataset,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        **kwargs: Any,
    ) -> None:
        sampler = ClassBalancedBatchSampler(
            dataset.labels,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

        kwargs.pop("batch_size", None)
        kwargs.pop("shuffle", None)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)
        loader_kwargs.update(kwargs)

        super().__init__(**loader_kwargs)

        self.balanced_sampler = sampler

    @property
    def num_classes(self) -> int:
        dataset: GenericImageDataset = self.dataset  # type: ignore[attr-defined]
        return dataset.num_classes

    @property
    def class_names(self) -> Sequence[str]:
        dataset: GenericImageDataset = self.dataset  # type: ignore[attr-defined]
        return dataset.class_names


# Backwards compatibility aliases
ImageNetDataset = GenericImageDataset
ImageNetBalancedDataLoader = GenericBalancedDataLoader
