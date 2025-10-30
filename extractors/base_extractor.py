"""Base extractor abstractions for compressing model parameters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from config import ConfigParser, ExtractorConfig

# from models import KMeans



logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """Base class for extracting and compressing model parameters."""

    def __init__(self, name: str) -> None:
        self.config = ConfigParser.get(ExtractorConfig)
        self.name = name
        self.output_stack_size = self.config.output_stack_size

    @abstractmethod
    def load_parameters(self) -> list[torch.Tensor]:
        """Return model weights and biases as a NumPy array."""

    def extract(self) -> torch.Tensor:
        """Cluster each parameter column down to a fixed number of centers."""
        matrix = self.load_parameters()
        matrix.reverse()

        self.cluster_count = self.config.output_stack_size // len(matrix)

        if len(matrix) == 0:
            raise ValueError("No parameters found to extract.")

        clustered_columns = [
            self.k_means_clustering(column, self.cluster_count)
            if len(column) >= self.cluster_count else column.flatten()
            for column in matrix
        ]

        # clustered_columns = KMeans(n_clusters=self.cluster_count).fit(matrix)
        output = self.output_transform(clustered_columns)
        logger.info(f"Extracted output shape: {output.shape}, dtype: {output.dtype}, device: {output.device}")
        return output


    def output_transform(self, clustered_columns: Sequence[np.ndarray]) -> np.ndarray:
        """
        Make the output is of fixed size output_stack_size by padding or truncating.
        """
        concatenated = np.concatenate(clustered_columns)
        current_size = concatenated.size

        if current_size < self.output_stack_size:
            pad_amount = self.output_stack_size - current_size
            concatenated = np.pad(concatenated, (0, pad_amount))
        elif current_size > self.output_stack_size:
            concatenated = concatenated.narrow(0, 0, self.output_stack_size)
        return concatenated

    def k_means_clustering(self, values: Sequence[float] | np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Cluster scalar values using scikit-learn's KMeans implementation.
        """
        data = np.asarray(values, dtype=float).reshape(-1, 1)
        if data.size == 0:
            raise ValueError("k_means_clustering requires at least one value.")

        model = KMeans(n_clusters=n_clusters, n_init="auto").fit(data)
        centers = np.sort(model.cluster_centers_.ravel())[::-1]
        return centers[: n_clusters]

    def save(self, data: Iterable[Iterable[float]] | np.ndarray) -> Path:
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
        return output_path
