"""Datasets and loaders for model/dataset similarity training."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterator, Sequence, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (ConfigParser, DatasetTokenLoaderConfig,
                    ModelEmbeddingLoaderConfig)


class ModelEmbeddingItem(TypedDict):
    model_token: torch.Tensor
    model_name: str
    model_index: int


class ModelEmbeddingBatch(TypedDict):
    model_tokens: torch.Tensor
    model_names: list[str]
    model_indices: torch.Tensor


class DatasetTokenItemRequired(TypedDict):
    dataset_name: str
    split: str
    dataset_tokens: torch.Tensor


class DatasetTokenItem(DatasetTokenItemRequired, total=False):
    class_ids: torch.Tensor
    class_names: list[list[str]]
    true_ranks: torch.Tensor


class DatasetTokenBatchRequired(TypedDict):
    dataset_name: list[str]
    split: list[str]
    dataset_tokens: torch.Tensor


class DatasetTokenBatch(DatasetTokenBatchRequired, total=False):
    class_ids: torch.Tensor
    class_names: list[list[str]]
    true_ranks: torch.Tensor


def _load_npz_embedding(path: Path, key: str) -> np.ndarray:
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"Expected key '{key}' in {path}")
        return archive[key]


class ModelEmbeddingDataset(Dataset[ModelEmbeddingItem]):
    """Dataset that yields pre-computed model embeddings and metadata."""

    def __init__(self, *, max_models: int | None = None) -> None:
        self.cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
        self.root_dir = Path(self.cfg.root_dir).expanduser()
        self.embedding_key = self.cfg.embedding_key
        self.dtype = torch.float32

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Model embeddings directory not found: {self.root_dir}")

        files = sorted(f for f in self.root_dir.rglob("*.npz") if f.is_file())
        if not files:
            raise ValueError(f"No embedding archives found under {self.root_dir}")

        max_allowed = max_models if max_models is not None else getattr(self.cfg, "max_models", None)
        if max_allowed is not None and max_allowed > 0:
            files = files[: int(max_allowed)]

        self.files: list[Path] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> ModelEmbeddingItem:
        file_path = self.files[index]
        embedding = _load_npz_embedding(file_path, self.embedding_key)
        tensor = torch.as_tensor(embedding, dtype=self.dtype)
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        return {"model_token": tensor, "model_name": file_path.stem, "model_index": index}


def build_model_embedding_loader(*, max_models: int | None = None, shuffle: bool | None = None) -> DataLoader:
    dataset = ModelEmbeddingDataset(max_models=max_models)
    effective_shuffle = dataset.cfg.shuffle if shuffle is None else shuffle

    def _collate(batch: Sequence[ModelEmbeddingItem]) -> ModelEmbeddingBatch:
        model_tokens = torch.stack([item["model_token"] for item in batch], dim=0)
        names = [str(item["model_name"]) for item in batch]
        indices = torch.tensor([int(item["model_index"]) for item in batch], dtype=torch.long)
        return {"model_tokens": model_tokens, "model_names": names, "model_indices": indices}

    return DataLoader(
        dataset,
        batch_size=dataset.cfg.batch_size,
        shuffle=effective_shuffle,
        num_workers=dataset.cfg.num_workers,
        pin_memory=dataset.cfg.pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )


@dataclass
class DatasetShardEntry:
    dataset_name: str
    split: str
    shard_path: Path


class DatasetTokenDataset(Dataset[DatasetTokenItem]):
    """Dataset that yields class-level token representations from shard files."""

    def __init__(
        self,
        *,
        splits: Sequence[str] | None = None,
        dataset_names: Sequence[str] | None = None,
        max_shards: int | None = None,
    ) -> None:
        self.cfg = ConfigParser.get(DatasetTokenLoaderConfig)
        self.root_dir = Path(self.cfg.root_dir).expanduser()
        self.dtype = torch.float32

        configured_names = list(self.cfg.dataset_names) if self.cfg.dataset_names is not None else None
        self.dataset_names = (
            list(dataset_names) if dataset_names is not None else configured_names
        )
        configured_splits = list(self.cfg.splits)
        self.splits = list(splits) if splits is not None else configured_splits
        self.shard_glob = self.cfg.shard_glob
        self.include_class_metadata = self.cfg.include_class_metadata
        self.max_batches = self.cfg.batch_size
        self.max_shards = max_shards if max_shards is not None else getattr(self.cfg, "max_shards", None)

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
        if self.max_shards is not None and self.max_shards > 0:
            entries = entries[: int(self.max_shards)]

        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DatasetTokenItem:
        entry = self.entries[index]
        with np.load(entry.shard_path, allow_pickle=True) as archive:
            features = archive["features"]
            class_ids = archive.get("class_ids")
            class_names = archive.get("class_names")
            actual_batches = archive.get("actual_batches")

        if features.ndim != 3:
            raise ValueError(f"Expected 3D features tensor in {entry.shard_path}, found shape {features.shape}")

        batch_limit = min(features.shape[0], self.max_batches)
        features = features[:batch_limit]
        token_tensor = torch.as_tensor(features, dtype=self.dtype)

        payload: DatasetTokenItem = {
            "dataset_name": entry.dataset_name,
            "split": entry.split,
            "dataset_tokens": token_tensor,
        }

        if self.include_class_metadata:
            payload["class_ids"] = torch.as_tensor(class_ids[:batch_limit]) if class_ids is not None else None
            payload["class_names"] = class_names[:batch_limit] if class_names is not None else None

        return payload


def build_dataset_token_loader(
    *,
    splits: Sequence[str] | None = None,
    shuffle: bool | None = None,
    max_shards: int | None = None,
    dataset_names: Sequence[str] | None = None,
) -> DataLoader:
    dataset = DatasetTokenDataset(splits=splits, dataset_names=dataset_names, max_shards=max_shards)
    effective_shuffle = dataset.cfg.shuffle if shuffle is None else shuffle

    def _collate(batch: Sequence[DatasetTokenItem]) -> DatasetTokenBatch:
        # will always recieve only one DatasetTokenItem from each shard, which already has batches
        dataset_token_item = batch[0]
        return DatasetTokenBatch(**dataset_token_item)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=effective_shuffle,
        num_workers=dataset.cfg.num_workers,
        pin_memory=dataset.cfg.pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )
