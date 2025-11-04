"""Datasets and loaders for model/dataset similarity training."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ConfigParser, DatasetTokenLoaderConfig, ModelEmbeddingLoaderConfig


def _load_npz_embedding(path: Path, key: str) -> np.ndarray:
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"Expected key '{key}' in {path}")
        return archive[key]


class ModelEmbeddingDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that yields pre-computed model embeddings and metadata."""

    def __init__(
        self,
        root_dir: Path | str | None = None,
        *,
        embedding_key: str = "embedding",
        max_models: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        ConfigParser.load()
        cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)

        self.root_dir = Path(root_dir or cfg.root_dir).expanduser()
        self.embedding_key = embedding_key or cfg.embedding_key
        self.dtype = dtype

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Model embeddings directory not found: {self.root_dir}")

        files = sorted(f for f in self.root_dir.rglob("*.npz") if f.is_file())
        if not files:
            raise ValueError(f"No embedding archives found under {self.root_dir}")

        if max_models is None:
            max_models = cfg.max_models

        if max_models is not None:
            files = files[: int(max_models)]

        self.files: list[Path] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        file_path = self.files[index]
        embedding = _load_npz_embedding(file_path, self.embedding_key)
        tensor = torch.as_tensor(embedding, dtype=self.dtype)
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        return {"embedding": tensor, "model_name": file_path.stem}


def build_model_embedding_loader(
    dataset: ModelEmbeddingDataset | None = None,
    *,
    config: ModelEmbeddingLoaderConfig | None = None,
) -> DataLoader:
    ConfigParser.load()
    cfg = config or ConfigParser.get(ModelEmbeddingLoaderConfig)
    dataset = dataset or ModelEmbeddingDataset(cfg.root_dir, embedding_key=cfg.embedding_key, max_models=cfg.max_models)

    def _collate(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
        embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
        names = [str(item["model_name"]) for item in batch]
        return {"embeddings": embeddings, "model_names": names}

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )


@dataclass
class DatasetShardEntry:
    dataset_name: str
    split: str
    shard_path: Path


class DatasetTokenDataset(Dataset[dict[str, torch.Tensor | list[str] | str]]):
    """Dataset that yields class-level token representations from shard files."""

    def __init__(
        self,
        root_dir: Path | str | None = None,
        *,
        dataset_names: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        shard_glob: str | None = None,
        average_over_batches: bool | None = None,
        include_class_metadata: bool | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        ConfigParser.load()
        cfg = ConfigParser.get(DatasetTokenLoaderConfig)

        self.root_dir = Path(root_dir or cfg.root_dir).expanduser()
        self.dtype = dtype

        names = dataset_names if dataset_names is not None else cfg.dataset_names
        self.dataset_names = list(names) if names is not None else None

        self.splits = list(splits) if splits is not None else list(cfg.splits)
        self.shard_glob = shard_glob or cfg.shard_glob
        self.average_over_batches = cfg.average_over_batches if average_over_batches is None else bool(average_over_batches)
        self.include_class_metadata = cfg.include_class_metadata if include_class_metadata is None else bool(include_class_metadata)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset embeddings directory not found: {self.root_dir}")

        entries: list[DatasetShardEntry] = []

        datasets = self.dataset_names or [p.name for p in self.root_dir.iterdir() if p.is_dir()]

        for dataset_name in sorted(datasets):
            dataset_dir = self.root_dir / dataset_name
            if not dataset_dir.exists():
                continue
            for split in self.splits:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                for shard_path in sorted(split_dir.glob(self.shard_glob)):
                    entries.append(DatasetShardEntry(dataset_name, split, shard_path))

        if not entries:
            raise ValueError(f"No dataset shards found under {self.root_dir}")

        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str] | str]:
        entry = self.entries[index]
        with np.load(entry.shard_path, allow_pickle=True) as archive:
            features = archive["features"]  # shape: [batches, num_classes, dim]
            class_ids = archive.get("class_ids")
            class_names = archive.get("class_names")

        if features.ndim != 3:
            raise ValueError(f"Expected 3D features tensor in {entry.shard_path}, found shape {features.shape}")

        if self.average_over_batches:
            tokens = features.mean(axis=0)  # [num_classes, dim]
        else:
            # Flatten batch dimension while preserving class grouping.
            num_batches, num_classes, dim = features.shape
            tokens = features.reshape(num_batches * num_classes, dim)

        token_tensor = torch.as_tensor(tokens, dtype=self.dtype)

        payload: dict[str, torch.Tensor | list[str] | str] = {
            "dataset_name": entry.dataset_name,
            "split": entry.split,
            "tokens": token_tensor,
        }

        if self.include_class_metadata:
            if class_ids is not None:
                payload["class_ids"] = torch.as_tensor(class_ids)
            if class_names is not None:
                payload["class_names"] = [str(name) for name in class_names]

        return payload


def build_dataset_token_loader(
    dataset: DatasetTokenDataset | None = None,
    *,
    config: DatasetTokenLoaderConfig | None = None,
) -> DataLoader:
    ConfigParser.load()
    cfg = config or ConfigParser.get(DatasetTokenLoaderConfig)
    dataset = dataset or DatasetTokenDataset(
        cfg.root_dir,
        dataset_names=cfg.dataset_names,
        splits=cfg.splits,
        shard_glob=cfg.shard_glob,
        average_over_batches=cfg.average_over_batches,
        include_class_metadata=cfg.include_class_metadata,
    )

    def _collate(batch: Sequence[dict[str, torch.Tensor | list[str] | str]]) -> dict[str, torch.Tensor | list[str] | list[str]]:
        # Expect batch_size=1 for simplicity; otherwise return list.
        if len(batch) == 1:
            return batch[0]
        return {
            "dataset_name": [item["dataset_name"] for item in batch],
            "split": [item["split"] for item in batch],
            "tokens": [item["tokens"] for item in batch],
            "class_ids": [item.get("class_ids") for item in batch],
            "class_names": [item.get("class_names") for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )


def cycle_model_batches(loader: DataLoader) -> Iterator[dict[str, torch.Tensor | list[str]]]:
    """Convenience iterator that cycles through model batches indefinitely."""

    for batch in cycle(loader):
        yield batch
