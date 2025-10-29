"""Base extractor abstractions for compressing model parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from keras.utils import set_random_seed as keras_set_random_seed
except ImportError:  # pragma: no cover - optional dependency
    keras_set_random_seed = None  # type: ignore[assignment]

from config import ConfigParser, ExtractorConfig


class BaseExtractor(ABC):
    """Base class for extracting and compressing model parameters."""

    def __init__(self, name: str) -> None:
        self.config = ConfigParser.get(ExtractorConfig)
        self.name = name

    @abstractmethod
    def load_parameters(self) -> np.ndarray:
        """Return model weights and biases as a NumPy array."""

    def extract(self) -> list[list[float]]:
        """Cluster each parameter column down to a fixed number of centers."""
        matrix = np.asarray(self.load_parameters(), dtype=float)
        if matrix.size == 0:
            raise ValueError("No parameters found to extract.")
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        clustered_columns = [
            self.k_means_clustering(matrix[:, idx]) for idx in range(matrix.shape[1])
        ]
        return np.stack(clustered_columns, axis=1).tolist()

    def k_means_clustering(
        self,
        values: Sequence[float] | np.ndarray,
        n_clusters: int | None = None,
    ) -> np.ndarray:
        """Cluster scalar values using scikit-learn's KMeans implementation."""
        data = np.asarray(values, dtype=float).reshape(-1, 1)
        if data.size == 0:
            raise ValueError("k_means_clustering requires at least one value.")

        model = KMeans(
            n_clusters=n_clusters or self.config.cluster_count,
            n_init="auto",
        ).fit(data)
        centers = np.sort(model.cluster_centers_.ravel())[::-1]
        return centers[: self.config.cluster_count]

    def save(
        self,
        data: Iterable[Iterable[float]] | np.ndarray,
    ) -> Path:
        """Save extracted parameters to a compressed .npz file."""
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        target_dir = Path(self.config.model_output_folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{self.name}.npz"
        output_path = target_dir / output_name
        np.savez_compressed(output_path, parameters=array)
        return output_path
