"""Combined dataloader for similarity transformer training with integrated model and dataset tokens.

This module provides a unified dataloader that combines model embeddings and dataset tokens
into structured batches suitable for training the CustomSimilarityTransformer model.

Key Features:
- Loads pre-computed model embeddings and dataset tokens from disk
- Constructs batches containing all necessary information for similarity learning
- Handles variable-length dataset tokens and fixed model count
- Supports train/val/test splits
- Provides ground truth ranking labels from performance JSON
- Integrates model names, indices, and dataset names for tracking and analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TypedDict

import numpy as np
import torch
from beautilog import logger
from torch.utils.data import DataLoader, Dataset

from config import ConfigParser, DatasetTokenLoaderConfig, ModelEmbeddingLoaderConfig


class CombinedSimilarityItem(TypedDict):
    """Single item from combined similarity dataset."""

    dataset_name: str
    split: str
    dataset_tokens: torch.Tensor  # Shape: (M, D) - M class batches, D-dim embeddings
    model_tokens: torch.Tensor  # Shape: (N, D) - N models, D-dim embeddings
    model_names: list[str]  # Length N
    model_indices: torch.Tensor  # Shape: (N,) - indices of models
    true_ranks: torch.Tensor  # Shape: (N,) - ranking scores or indices
    metadata: dict[str, str | int]  # Extra tracking info


class CombinedSimilarityBatch(TypedDict, total=False):
    """Batch from combined similarity dataloader.

    This is the primary batch structure returned to trainers and is designed to
    contain all information needed for model-dataset similarity training.
    """

    # Core tensors required for model forward pass
    dataset_tokens: torch.Tensor  # Shape: (B, M, D) - batch of dataset token sequences
    model_tokens: torch.Tensor  # Shape: (B, N, D) - batch of model embeddings
    true_ranks: torch.Tensor  # Shape: (B, N) - batch of ground truth rankings

    # Metadata for tracking and analysis
    model_names: list[list[str]]  # Length B, each sublist length N
    model_indices: torch.Tensor  # Shape: (B, N)
    dataset_names: list[str]  # Length B
    dataset_splits: list[str]  # Length B
    batch_size: int


def _load_npz_embedding(path: Path, key: str) -> np.ndarray:
    """Load embedding from NPZ archive with error handling.

    Args:
        path: Path to NPZ file
        key: Key to extract from archive

    Returns:
        NumPy array containing embedding data

    Raises:
        KeyError: If key not found in archive
        FileNotFoundError: If file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"Expected key '{key}' in {path}, found keys: {list(archive.keys())}")
        return archive[key]


def _load_performance_rankings(
    perf_json_path: Path | None, dataset_name: str, model_names: list[str]
) -> np.ndarray | None:
    """Load ground truth performance rankings for a dataset-model combination.

    Ranks are derived from model accuracy scores. Higher accuracy = higher rank (1.0).
    Models not found in performance data are assigned a rank of 0.5.

    Args:
        perf_json_path: Path to performance JSON file (e.g., dataset_model_performance.json)
        dataset_name: Name of dataset to look up
        model_names: List of N model names to rank

    Returns:
        Array of shape (N,) with rank scores in [0, 1], or None if file/dataset not found

    Raises:
        JSONDecodeError: If JSON file is malformed
    """
    if perf_json_path is None or not perf_json_path.exists():
        logger.debug("Performance JSON file not found: %s", perf_json_path)
        return None

    try:
        with open(perf_json_path) as f:
            perf_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load performance rankings from %s: %s", perf_json_path, e)
        return None

    if dataset_name not in perf_data:
        logger.debug("Dataset '%s' not found in performance data", dataset_name)
        return None

    dataset_perfs = perf_data[dataset_name]
    perf_dict = {item["model_name"]: item["accuracy"] for item in dataset_perfs}

    # Normalize accuracies to [0, 1] range (most likely already normalized)
    accuracies = np.array(
        [perf_dict.get(name, 0.5) for name in model_names], dtype=np.float32
    )
    ranks = np.clip(accuracies, 0.0, 1.0)

    logger.debug(
        "Loaded %d/%d model rankings for dataset '%s'",
        sum(name in perf_dict for name in model_names),
        len(model_names),
        dataset_name,
    )

    return ranks


@dataclass
class ModelEmbeddingEntry:
    """Represents a single model embedding file."""

    model_name: str
    model_index: int
    embedding_path: Path


@dataclass
class DatasetTokenEntry:
    """Represents a single dataset token shard."""

    dataset_name: str
    split: str
    shard_path: Path


class CombinedSimilarityDataset(Dataset[CombinedSimilarityItem]):
    """Dataset that combines model embeddings and dataset tokens into unified batches.

    This dataset:
    1. Loads all available model embeddings (fixed set)
    2. Loads dataset tokens for each dataset/split combination
    3. Combines them with ground truth rankings to create training items
    4. Returns structured items ready for batch collation

    Attributes:
        model_embeddings_dir: Directory containing model embedding NPZ files
        dataset_tokens_dir: Directory containing dataset token shard NPZ files
        performance_json_path: Optional path to ground truth performance rankings
        max_models: Maximum number of models to include (None = all)
        max_dataset_shards: Maximum number of dataset shards to include (None = all)
        dtype: PyTorch dtype for tensors (default: float32)
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
        """Initialize combined similarity dataset.

        Args:
            model_embeddings_dir: Path to directory with model embedding NPZ files.
                If None, uses config from ModelEmbeddingLoaderConfig.
            dataset_tokens_dir: Path to directory with dataset token shards.
                If None, uses config from DatasetTokenLoaderConfig.
            performance_json_path: Path to JSON file with ground truth rankings.
                Can be absolute or relative to project root.
            splits: Dataset splits to include (e.g., ["train", "validation", "test"]).
                If None, uses config defaults.
            dataset_names: Specific dataset names to include.
                If None, includes all found datasets.
            max_models: Maximum number of models to load (None = all).
            max_dataset_shards: Maximum number of dataset shards to load (None = all).

        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If no embeddings or dataset tokens found
        """
        self.dtype = torch.float32

        # Load configuration
        model_cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
        dataset_cfg = ConfigParser.get(DatasetTokenLoaderConfig)

        # Resolve model embeddings directory
        if model_embeddings_dir is None:
            model_embeddings_dir = model_cfg.root_dir
        self.model_embeddings_dir = Path(model_embeddings_dir).expanduser()

        # Resolve dataset tokens directory
        if dataset_tokens_dir is None:
            dataset_tokens_dir = dataset_cfg.root_dir
        self.dataset_tokens_dir = Path(dataset_tokens_dir).expanduser()

        # Resolve performance JSON path
        if performance_json_path is not None:
            perf_path = Path(performance_json_path).expanduser()
            if not perf_path.is_absolute():
                perf_path = Path.cwd() / perf_path
            self.performance_json_path = perf_path
        else:
            # Try to find it in project constants
            project_root = Path.cwd()
            default_perf = project_root / "constants" / "dataset_model_performance.json"
            self.performance_json_path = default_perf if default_perf.exists() else None

        # Validate directories
        if not self.model_embeddings_dir.exists():
            raise FileNotFoundError(f"Model embeddings directory not found: {self.model_embeddings_dir}")
        if not self.dataset_tokens_dir.exists():
            raise FileNotFoundError(f"Dataset tokens directory not found: {self.dataset_tokens_dir}")

        # Store configuration
        self.embedding_key = model_cfg.embedding_key
        self.shard_glob = dataset_cfg.shard_glob
        self.max_models = max_models
        self.max_dataset_shards = max_dataset_shards
        self.include_class_metadata = dataset_cfg.include_class_metadata

        # Load model embeddings
        self.model_entries = self._load_model_embeddings()
        if not self.model_entries:
            raise ValueError(f"No model embeddings found in {self.model_embeddings_dir}")

        logger.info("Loaded %d model embeddings from %s", len(self.model_entries), self.model_embeddings_dir)

        # Load dataset token entries
        configured_splits = splits or tuple(dataset_cfg.splits)
        configured_names = dataset_names
        self.dataset_entries = self._load_dataset_entries(configured_splits, configured_names)
        if not self.dataset_entries:
            raise ValueError(f"No dataset tokens found in {self.dataset_tokens_dir}")

        logger.info(
            "Loaded %d dataset token shards from %s (%d unique datasets)",
            len(self.dataset_entries),
            self.dataset_tokens_dir,
            len(set(entry.dataset_name for entry in self.dataset_entries)),
        )

    def _load_model_embeddings(self) -> list[ModelEmbeddingEntry]:
        """Discover and catalog all available model embedding files.

        Returns:
            Sorted list of ModelEmbeddingEntry objects
        """
        entries = []
        files = sorted(f for f in self.model_embeddings_dir.rglob("*.npz") if f.is_file())

        if self.max_models is not None and self.max_models > 0:
            files = files[: int(self.max_models)]

        for idx, fpath in enumerate(files):
            entries.append(
                ModelEmbeddingEntry(
                    model_name=fpath.stem,
                    model_index=idx,
                    embedding_path=fpath,
                )
            )

        return entries

    def _load_dataset_entries(
        self, splits: Sequence[str], dataset_names: Sequence[str] | None
    ) -> list[DatasetTokenEntry]:
        """Discover and catalog all available dataset token shards.

        Args:
            splits: Which splits to include
            dataset_names: Specific dataset names to include, or None for all

        Returns:
            Sorted list of DatasetTokenEntry objects
        """
        entries = []

        # Determine which datasets to scan
        if dataset_names is not None:
            datasets_to_scan = list(dataset_names)
        else:
            datasets_to_scan = [
                p.name for p in self.dataset_tokens_dir.iterdir() if p.is_dir()
            ]

        for dataset_name in sorted(datasets_to_scan):
            dataset_dir = self.dataset_tokens_dir / dataset_name
            if not dataset_dir.exists():
                logger.warning("Dataset directory not found: %s", dataset_dir)
                continue

            for split in splits:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue

                for shard_path in sorted(split_dir.glob(self.shard_glob)):
                    entries.append(
                        DatasetTokenEntry(
                            dataset_name=dataset_name,
                            split=split,
                            shard_path=shard_path,
                        )
                    )

        if self.max_dataset_shards is not None and self.max_dataset_shards > 0:
            entries = entries[: int(self.max_dataset_shards)]

        return entries

    def __len__(self) -> int:
        """Return total number of dataset-split combinations."""
        return len(self.dataset_entries)

    def __getitem__(self, index: int) -> CombinedSimilarityItem:
        """Get a combined similarity item with all model and dataset information.

        Args:
            index: Index into dataset_entries

        Returns:
            Dictionary containing all tensors and metadata for a single dataset

        Raises:
            ValueError: If shapes are invalid or data is corrupted
            RuntimeError: If true_ranks cannot be computed
        """
        dataset_entry = self.dataset_entries[index]

        # Load dataset tokens
        dataset_tokens = self._load_dataset_tokens(dataset_entry.shard_path)

        # Load all model embeddings
        model_tokens, model_names, model_indices = self._load_all_model_embeddings()

        # Compute true ranking labels
        true_ranks = self._compute_true_ranks(dataset_entry.dataset_name, model_names)

        return CombinedSimilarityItem(
            dataset_name=dataset_entry.dataset_name,
            split=dataset_entry.split,
            dataset_tokens=dataset_tokens,
            model_tokens=model_tokens,
            model_names=model_names,
            model_indices=model_indices,
            true_ranks=true_ranks,
            metadata={
                "shard_path": str(dataset_entry.shard_path),
                "num_dataset_batches": dataset_tokens.shape[0],
                "num_models": len(model_names),
            },
        )

    def _load_dataset_tokens(self, shard_path: Path) -> torch.Tensor:
        """Load and validate dataset token tensor from shard file.

        Args:
            shard_path: Path to NPZ shard file

        Returns:
            Tensor of shape (M, D) - M class batches, D-dim embeddings

        Raises:
            ValueError: If shape is invalid or required keys missing
        """
        with np.load(shard_path, allow_pickle=True) as archive:
            if "features" not in archive:
                raise ValueError(f"'features' key missing in {shard_path}")
            features = archive["features"]

        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D dataset features, got shape {features.shape} in {shard_path}"
            )

        # features shape: (batches, num_classes, embedding_dim)
        # Stack into (num_classes, embedding_dim) by taking first batch
        # or flattening if single batch provided
        if features.shape[0] == 1:
            tokens = features[0]  # Shape: (num_classes, dim)
        else:
            # Concatenate all batches along class dimension
            tokens = np.concatenate([features[i] for i in range(features.shape[0])], axis=0)

        tensor = torch.as_tensor(tokens, dtype=self.dtype)

        if tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D token tensor after processing, got {tensor.ndim}D "
                f"with shape {tensor.shape}"
            )

        return tensor

    def _load_all_model_embeddings(self) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Load all model embeddings into stacked tensors.

        Returns:
            Tuple of:
            - model_tokens: Shape (N, D) - N models, D-dim embeddings
            - model_names: List of length N
            - model_indices: Shape (N,) - indices of models

        Raises:
            ValueError: If embeddings have inconsistent shapes
        """
        model_embeddings = []
        model_names = []
        model_indices = []

        for entry in self.model_entries:
            embedding = _load_npz_embedding(entry.embedding_path, self.embedding_key)
            tensor = torch.as_tensor(embedding, dtype=self.dtype)

            # Flatten if needed
            if tensor.ndim > 1:
                tensor = tensor.reshape(-1)

            model_embeddings.append(tensor)
            model_names.append(entry.model_name)
            model_indices.append(entry.model_index)

        # Validate all embeddings have same dimension
        dims = [t.shape[0] for t in model_embeddings]
        if len(set(dims)) > 1:
            logger.warning(
                "Model embeddings have inconsistent dimensions: %s. "
                "This may cause issues.",
                set(dims),
            )

        # Stack into tensor
        model_tokens = torch.stack(model_embeddings, dim=0)
        model_indices_tensor = torch.tensor(model_indices, dtype=torch.long)

        return model_tokens, model_names, model_indices_tensor

    def _compute_true_ranks(self, dataset_name: str, model_names: list[str]) -> torch.Tensor:
        """Compute ground truth ranking scores for models on a dataset.

        Args:
            dataset_name: Name of dataset
            model_names: List of model names to rank

        Returns:
            Tensor of shape (N,) with ranking scores in [0, 1]
        """
        ranks = _load_performance_rankings(self.performance_json_path, dataset_name, model_names)

        if ranks is None:
            logger.debug(
                "No ground truth rankings for dataset '%s', using uniform scores",
                dataset_name,
            )
            ranks = np.ones(len(model_names), dtype=np.float32) / len(model_names)

        return torch.as_tensor(ranks, dtype=self.dtype)


def _collate_combined_similarity(
    batch: Sequence[CombinedSimilarityItem],
) -> CombinedSimilarityBatch:
    """Collate function for combining similarity items into a batch.

    Handles variable-length dataset tokens by stacking model tokens uniformly.
    Dataset tokens remain as separate tensors since they have variable batch lengths.

    Args:
        batch: List of CombinedSimilarityItem dicts

    Returns:
        CombinedSimilarityBatch with tensors stacked/padded appropriately

    Raises:
        ValueError: If batch is empty or has inconsistent model counts
    """
    if not batch:
        raise ValueError("Empty batch passed to collate function")

    batch_size = len(batch)

    # Check consistency: all items must have same number of models
    num_models_list = [item["model_tokens"].shape[0] for item in batch]
    if len(set(num_models_list)) > 1:
        raise ValueError(
            f"Inconsistent model counts in batch: {set(num_models_list)}. "
            "All items must have same number of models."
        )

    # Stack model tokens: (B, N, D)
    model_tokens = torch.stack([item["model_tokens"] for item in batch], dim=0)

    # Stack model indices: (B, N)
    model_indices = torch.stack([item["model_indices"] for item in batch], dim=0)

    # Stack true ranks: (B, N)
    true_ranks = torch.stack([item["true_ranks"] for item in batch], dim=0)

    # Dataset tokens have variable batch lengths, so keep as list
    # But we need to stack them for the model. Since they can vary, we'll pad.
    dataset_tokens_list = [item["dataset_tokens"] for item in batch]

    # Find max number of classes across batch
    max_classes = max(dt.shape[0] for dt in dataset_tokens_list)
    embed_dim = dataset_tokens_list[0].shape[1]

    # Pad all dataset tokens to max_classes
    padded_dataset_tokens = []
    for dt in dataset_tokens_list:
        if dt.shape[0] < max_classes:
            padding = torch.zeros(
                max_classes - dt.shape[0], embed_dim, dtype=dt.dtype, device=dt.device
            )
            dt_padded = torch.cat([dt, padding], dim=0)
        else:
            dt_padded = dt
        padded_dataset_tokens.append(dt_padded)

    # Stack into (B, M, D) where M is max_classes
    dataset_tokens = torch.stack(padded_dataset_tokens, dim=0)

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
    """Build a DataLoader for combined similarity transformer training.

    Args:
        model_embeddings_dir: Directory containing model embedding NPZ files.
            If None, uses ModelEmbeddingLoaderConfig.
        dataset_tokens_dir: Directory containing dataset token shards.
            If None, uses DatasetTokenLoaderConfig.
        performance_json_path: Path to JSON file with ground truth rankings.
            If None, attempts to find at constants/dataset_model_performance.json.
        splits: Dataset splits to include (default: all splits from config).
        dataset_names: Specific dataset names to include (default: all found).
        batch_size: Batch size for DataLoader (default: from config).
        shuffle: Whether to shuffle dataset (default: True).
        max_models: Maximum number of models to load (default: all).
        max_dataset_shards: Maximum number of dataset shards (default: all).
        num_workers: Number of worker processes (default: from config).
        pin_memory: Whether to pin memory (default: from config).

    Returns:
        DataLoader yielding CombinedSimilarityBatch items

    Example:
        >>> loader = build_combined_similarity_loader(
        ...     splits=["train"],
        ...     shuffle=True,
        ...     batch_size=16,
        ... )
        >>> for batch in loader:
        ...     dataset_tokens = batch["dataset_tokens"]  # (B, M, D)
        ...     model_tokens = batch["model_tokens"]      # (B, N, D)
        ...     true_ranks = batch["true_ranks"]          # (B, N)
    """
    # Load configuration defaults
    model_cfg = ConfigParser.get(ModelEmbeddingLoaderConfig)
    dataset_cfg = ConfigParser.get(DatasetTokenLoaderConfig)

    # Create dataset
    dataset = CombinedSimilarityDataset(
        model_embeddings_dir=model_embeddings_dir,
        dataset_tokens_dir=dataset_tokens_dir,
        performance_json_path=performance_json_path,
        splits=splits,
        dataset_names=dataset_names,
        max_models=max_models,
        max_dataset_shards=max_dataset_shards,
    )

    # Use provided values or config defaults
    effective_batch_size = batch_size if batch_size is not None else dataset_cfg.batch_size
    effective_shuffle = shuffle if shuffle is not None else dataset_cfg.shuffle
    effective_num_workers = num_workers if num_workers is not None else dataset_cfg.num_workers
    effective_pin_memory = pin_memory if pin_memory is not None else dataset_cfg.pin_memory

    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=effective_shuffle,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        collate_fn=_collate_combined_similarity,
        drop_last=False,
    )
