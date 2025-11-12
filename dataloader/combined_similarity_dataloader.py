"""Unified dataloader for similarity transformer training combining model and dataset embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence, TypedDict

import numpy as np
import torch
from beautilog import logger
from torch.utils.data import DataLoader, Dataset

from config import (ConfigParser, DatasetTokenLoaderConfig,
                    ModelEmbeddingLoaderConfig)

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

    def __init__(self,split: str) -> None:
        """Initialize dataset with pre-loaded model embeddings and dataset token catalog.

        Args:
            split: Dataset split to include (default: from config)
        """
        self.dtype = torch.float32

        # Load configuration
        self.model_cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
        self.dataset_cfg = ConfigParser.get(DatasetTokenLoaderConfig)

        # Resolve directories
        self.model_embeddings_dir = Path(self.model_cfg.root_dir).expanduser()
        self.dataset_tokens_dir = Path(self.dataset_cfg.root_dir).expanduser()

        # Resolve performance JSON
        self.perf_json_path = Path.cwd() / "constants" / "dataset_model_performance.json"

        # Validate directories
        if not self.model_embeddings_dir.exists():
            raise FileNotFoundError(f"Model embeddings dir not found: {self.model_embeddings_dir}")
        if not self.dataset_tokens_dir.exists():
            raise FileNotFoundError(f"Dataset tokens dir not found: {self.dataset_tokens_dir}")

        # Store config
        self.embedding_key = self.model_cfg.embedding_key
        self.shard_glob = self.dataset_cfg.shard_glob

        # Pre-load all model embeddings (fixed set, reused across all samples)
        self.model_tokens, self.model_names, self.model_indices = self._load_all_models(self.model_cfg.max_models)
        if not self.model_names:
            raise ValueError(f"No model embeddings found in {self.model_embeddings_dir}")

        logger.info("Loaded %d model embeddings", len(self.model_names))

        # Catalog dataset entries (load on-demand per sample)
        self.split = split
        self.dataset_entries = self._catalog_dataset_entries(split, self.dataset_cfg.dataset_names, self.dataset_cfg.max_shards)

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
        self, split: str, dataset_names: Sequence[str] | None, max_shards: int | None
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

            split_dir = dataset_dir / split
            if not split_dir.exists():
                logging.warning("Split directory not found for dataset %s: %s", dataset_name, split_dir)
                continue

            for shard_path in sorted(split_dir.glob(self.shard_glob)):
                entries.append((dataset_name, shard_path))

        if max_shards and max_shards > 0:
            entries = entries[:max_shards]

        return entries

    def __len__(self) -> int:
        return len(self.dataset_entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get item with pre-loaded model embeddings and on-demand dataset tokens."""
        dataset_name, shard_path = self.dataset_entries[index]

        # Sample models for this dataset
        models = self.model_names[: self.model_cfg.max_models] if self.model_cfg.max_models else self.model_names
        model_names = models[: self.model_cfg.max_models] if self.model_cfg.max_models else models
        model_indices = self.model_indices[: self.model_cfg.max_models] if self.model_cfg.max_models else self.model_indices
        if self.model_cfg.shuffle:
            perm = torch.randperm(len(models))
            models = [models[i] for i in perm]
            model_names = [model_names[i] for i in perm]
            model_indices = model_indices[perm]

        # Load dataset tokens from shard
        dataset_tokens = self._load_dataset_tokens(shard_path)

        # Get true ranks for this dataset
        true_ranks = compute_true_ranks(dataset_name, model_names)

        return {
            "dataset_name": dataset_name,
            "dataset_tokens": dataset_tokens,
            "model_tokens": models,
            "model_names": model_names,
            "model_indices": model_indices,
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
        # tokens = np.concatenate([features[i] for i in range(features.shape[0])], axis=0)
        return features


def _collate_batch(batch: Sequence[dict]) -> CombinedSimilarityBatch:
    """Collate items into batch with padding for variable-length dataset tokens."""
    if not batch:
        raise ValueError("Empty batch")

    if len(batch) > 1:
        raise ValueError(f"Expected batch size of 1, got {len(batch)}")

    item = batch[0]

    return CombinedSimilarityBatch(
        dataset_tokens=item["dataset_tokens"],
        model_tokens=item["model_tokens"],
        true_ranks=item["true_ranks"],
        model_names=item["model_names"],
        model_indices=item["model_indices"],
        dataset_names=item["dataset_name"],
        batch_size=len(batch),
    )


def build_combined_similarity_loader(split: str) -> DataLoader:
    """Build DataLoader for combined similarity training.

    Args:
        split: Dataset split to include

    Returns:
        DataLoader yielding CombinedSimilarityBatch items
    """
    dataset = CombinedSimilarityDataset(split=split)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=dataset.dataset_cfg.shuffle,
        num_workers=dataset.dataset_cfg.num_workers,
        pin_memory=dataset.dataset_cfg.pin_memory,
        collate_fn=_collate_batch,
        drop_last=False,
    )
