"""Unified dataloader for similarity transformer training combining model and dataset embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TypedDict

import numpy as np
import torch
from beautilog import logger
from torch.utils.data import DataLoader, Dataset

from config import ConfigParser, DatasetTokenLoaderConfig, ModelEmbeddingLoaderConfig

from .ranking import compute_true_ranks

class CombinedSimilarityBatch(TypedDict, total=False):
    """Batch from combined similarity dataloader.

    Contains all information needed for model-dataset similarity training.
    """

    # Core tensors
    dataset_tokens: torch.Tensor  # Shape: (B, M, D) - batch of dataset token sequences
    model_tokens: torch.Tensor    # Shape: (B, N, D) - batch of model embeddings
    true_ranks: torch.Tensor      # Shape: (B, N) - batch of ground truth rankings

    # Metadata
    model_names: list[list[str]]  # Length B, each sublist length N
    model_indices: torch.Tensor   # Shape: (B, N)
    dataset_names: list[str]      # Length B
    dataset_splits: list[str]     # Length B
    batch_size: int


class CombinedSimilarityDataset(Dataset):
    """Dataset combining pre-loaded model embeddings with dataset token shards.

    Pre-loads all model embeddings at initialization since they're reused across
    all samples. Loads dataset tokens on-demand per sample for memory efficiency.
    """

    def __init__(
        self,
        *,
        model_embeddings_dir: Path | str | None = None,
        dataset_tokens_dir: Path | str | None = None,
        performance_json_path: Path | str | None = None,
        splits: Sequence[str] | None = None,
        dataset_names: Sequence[str] | None = None,
        max_models: int | None = None,
        max_dataset_shards: int | None = None,
    ) -> None:
        """Initialize dataset with pre-loaded model embeddings and dataset token catalog.

        Args:
            model_embeddings_dir: Directory with model embedding NPZ files
            dataset_tokens_dir: Directory with dataset token shard NPZ files
            performance_json_path: Path to ground truth performance rankings JSON
            splits: Dataset splits to include (default: from config)
            dataset_names: Specific dataset names to include (default: all)
            max_models: Maximum number of models to load
            max_dataset_shards: Maximum number of dataset shards to load
        """
        self.dtype = torch.float32

        # Load configuration
        model_cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
        dataset_cfg = ConfigParser.get(DatasetTokenLoaderConfig)

        # Resolve directories
        self.model_embeddings_dir = Path(model_embeddings_dir or model_cfg.root_dir).expanduser()
        self.dataset_tokens_dir = Path(dataset_tokens_dir or dataset_cfg.root_dir).expanduser()

        # Resolve performance JSON
        if performance_json_path is not None:
            perf_path = Path(performance_json_path).expanduser()
            if not perf_path.is_absolute():
                perf_path = Path.cwd() / perf_path
            self.perf_json_path = perf_path if perf_path.exists() else None
        else:
            default_perf = Path.cwd() / "constants" / "dataset_model_performance.json"
            self.perf_json_path = default_perf if default_perf.exists() else None

        # Validate directories
        if not self.model_embeddings_dir.exists():
            raise FileNotFoundError(f"Model embeddings dir not found: {self.model_embeddings_dir}")
        if not self.dataset_tokens_dir.exists():
            raise FileNotFoundError(f"Dataset tokens dir not found: {self.dataset_tokens_dir}")

        # Store config
        self.embedding_key = model_cfg.embedding_key
        self.shard_glob = dataset_cfg.shard_glob

        # Pre-load all model embeddings (fixed set, reused across all samples)
        self.model_tokens, self.model_names, self.model_indices = self._load_all_models(max_models)
        if not self.model_names:
            raise ValueError(f"No model embeddings found in {self.model_embeddings_dir}")

        logger.info("Loaded %d model embeddings", len(self.model_names))

        # Catalog dataset entries (load on-demand per sample)
        self.dataset_entries = self._catalog_dataset_entries(
            splits or dataset_cfg.splits, dataset_names, max_dataset_shards
        )
        if not self.dataset_entries:
            raise ValueError(f"No dataset tokens found in {self.dataset_tokens_dir}")

        logger.info(
            "Cataloged %d dataset shards (%d unique datasets)",
            len(self.dataset_entries),
            len(set(e[0] for e in self.dataset_entries)),
        )

    def _load_all_models(self, max_models: int | None) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Pre-load all model embeddings from disk.

        Returns:
            (model_tokens: (N, D), model_names: list of N, model_indices: (N,))
        """
        files = sorted(f for f in self.model_embeddings_dir.rglob("*.npz") if f.is_file())
        if max_models and max_models > 0:
            files = files[:max_models]

        embeddings, names, indices = [], [], []
        for idx, fpath in enumerate(files):
            try:
                with np.load(fpath) as archive:
                    emb = archive[self.embedding_key]
                tensor = torch.as_tensor(emb, dtype=self.dtype)
                if tensor.ndim > 1:
                    tensor = tensor.reshape(-1)
                embeddings.append(tensor)
                names.append(fpath.stem)
                indices.append(idx)
            except Exception as e:
                logger.warning("Failed to load model embedding %s: %s", fpath, e)
                continue

        return torch.stack(embeddings, dim=0), names, torch.tensor(indices, dtype=torch.long)

    def _catalog_dataset_entries(
        self, splits: Sequence[str], dataset_names: Sequence[str] | None, max_shards: int | None
    ) -> list[tuple[str, str, Path]]:
        """Catalog dataset token shard paths without loading.

        Returns:
            List of (dataset_name, split, shard_path) tuples
        """
        entries = []
        datasets = (
            list(dataset_names)
            if dataset_names
            else [p.name for p in self.dataset_tokens_dir.iterdir() if p.is_dir()]
        )

        for dataset_name in sorted(datasets):
            dataset_dir = self.dataset_tokens_dir / dataset_name
            if not dataset_dir.exists():
                continue
            for split in splits:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                for shard_path in sorted(split_dir.glob(self.shard_glob)):
                    entries.append((dataset_name, split, shard_path))

        if max_shards and max_shards > 0:
            entries = entries[:max_shards]

        return entries

    def __len__(self) -> int:
        return len(self.dataset_entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get item with pre-loaded model embeddings and on-demand dataset tokens."""
        dataset_name, split, shard_path = self.dataset_entries[index]

        # Load dataset tokens from shard
        dataset_tokens = self._load_dataset_tokens(shard_path)

        # Get true ranks for this dataset
        true_ranks = compute_true_ranks(dataset_name, self.model_names)

        return {
            "dataset_name": dataset_name,
            "split": split,
            "dataset_tokens": dataset_tokens,
            "model_tokens": self.model_tokens,
            "model_names": self.model_names,
            "model_indices": self.model_indices,
            "true_ranks": true_ranks,
        }

    def _load_dataset_tokens(self, shard_path: Path) -> torch.Tensor:
        """Load dataset token tensor from shard file.

        Returns:
            Tensor of shape (M, D) - M class batches, D-dim embeddings
        """
        with np.load(shard_path, allow_pickle=True) as archive:
            features = archive["features"]

        if features.ndim != 3:
            raise ValueError(f"Expected 3D features, got shape {features.shape}")

        # Stack batches: (batches, num_classes, dim) -> (num_classes, dim)
        tokens = np.concatenate([features[i] for i in range(features.shape[0])], axis=0)
        return torch.as_tensor(tokens, dtype=self.dtype)


def _collate_batch(batch: Sequence[dict]) -> CombinedSimilarityBatch:
    """Collate items into batch with padding for variable-length dataset tokens."""
    if not batch:
        raise ValueError("Empty batch")

    batch_size = len(batch)

    # Stack model tensors (same for all items)
    model_tokens = torch.stack([item["model_tokens"] for item in batch], dim=0)
    model_indices = torch.stack([item["model_indices"] for item in batch], dim=0)
    true_ranks = torch.stack([item["true_ranks"] for item in batch], dim=0)

    # Pad variable-length dataset tokens
    dataset_tokens_list = [item["dataset_tokens"] for item in batch]
    max_classes = max(dt.shape[0] for dt in dataset_tokens_list)
    embed_dim = dataset_tokens_list[0].shape[1]

    padded = []
    for dt in dataset_tokens_list:
        if dt.shape[0] < max_classes:
            padding = torch.zeros(max_classes - dt.shape[0], embed_dim, dtype=dt.dtype)
            dt = torch.cat([dt, padding], dim=0)
        padded.append(dt)

    dataset_tokens = torch.stack(padded, dim=0)

    return CombinedSimilarityBatch(
        dataset_tokens=dataset_tokens,
        model_tokens=model_tokens,
        true_ranks=true_ranks,
        model_names=[item["model_names"] for item in batch],
        model_indices=model_indices,
        dataset_names=[item["dataset_name"] for item in batch],
        dataset_splits=[item["split"] for item in batch],
        batch_size=batch_size,
    )


def build_combined_similarity_loader(
    *,
    model_embeddings_dir: Path | str | None = None,
    dataset_tokens_dir: Path | str | None = None,
    performance_json_path: Path | str | None = None,
    splits: Sequence[str] | None = None,
    dataset_names: Sequence[str] | None = None,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    max_models: int | None = None,
    max_dataset_shards: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build DataLoader for combined similarity training.

    Args:
        model_embeddings_dir: Directory with model embedding NPZ files
        dataset_tokens_dir: Directory with dataset token shard NPZ files
        performance_json_path: Path to ground truth performance rankings JSON
        splits: Dataset splits to include
        dataset_names: Specific dataset names to include
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle dataset
        max_models: Maximum number of models to load
        max_dataset_shards: Maximum number of dataset shards
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        DataLoader yielding CombinedSimilarityBatch items
    """
    model_cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
    dataset_cfg = ConfigParser.get(DatasetTokenLoaderConfig)

    dataset = CombinedSimilarityDataset(
        model_embeddings_dir=model_embeddings_dir,
        dataset_tokens_dir=dataset_tokens_dir,
        performance_json_path=performance_json_path,
        splits=splits,
        dataset_names=dataset_names,
        max_models=max_models,
        max_dataset_shards=max_dataset_shards,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else dataset_cfg.batch_size,
        shuffle=shuffle if shuffle is not None else dataset_cfg.shuffle,
        num_workers=num_workers if num_workers is not None else dataset_cfg.num_workers,
        pin_memory=pin_memory if pin_memory is not None else dataset_cfg.pin_memory,
        collate_fn=_collate_batch,
        drop_last=False,
    )
