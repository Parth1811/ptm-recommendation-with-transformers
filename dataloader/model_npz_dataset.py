"""Dataset helpers for loading extracted model parameter archives."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelParameterDataset(Dataset[torch.Tensor]):
    """Loads compressed parameter vectors saved by the extractor pipeline."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        flatten: bool = True,
        file_extensions: Iterable[str] | None = None,
        normalize: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Parameter directory not found: {self.root_dir}")

        self.dtype = dtype
        self.flatten = flatten
        self.normalize = normalize
        self.epsilon = float(epsilon)
        extensions = tuple(file_extensions) if file_extensions else (".npz",)
        self.files = sorted(
            path
            for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix in extensions
        )
        if not self.files:
            raise ValueError(f"No parameter archives found in {self.root_dir}")

        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None

        if self.normalize:
            self._compute_normalization_stats()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.files[index]
        with np.load(file_path) as archive:
            parameters = archive["parameters"]

        if self.flatten:
            parameters = parameters.reshape(-1)

        tensor = torch.as_tensor(parameters, dtype=self.dtype)

        if self.normalize and self._feature_mean is not None and self._feature_std is not None:
            tensor = (tensor - self._feature_mean) / self._feature_std

        return tensor

    def _compute_normalization_stats(self) -> None:
        """Compute per-feature mean and std using a single pass over the dataset."""
        sum_: torch.Tensor | None = None
        sum_sq: torch.Tensor | None = None
        count = 0

        for file_path in self.files:
            with np.load(file_path) as archive:
                parameters = archive["parameters"]

            if self.flatten:
                parameters = parameters.reshape(-1)

            tensor = torch.as_tensor(parameters, dtype=torch.float64)

            if sum_ is None or sum_sq is None:
                sum_ = torch.zeros_like(tensor)
                sum_sq = torch.zeros_like(tensor)

            sum_ += tensor
            sum_sq += tensor * tensor
            count += 1

        if sum_ is None or sum_sq is None or count == 0:
            raise RuntimeError("Failed to compute normalization statistics; dataset appears to be empty.")

        mean = sum_ / count
        mean_sq = sum_sq / count
        variance = torch.clamp(mean_sq - mean.pow(2), min=self.epsilon ** 2)
        std = torch.sqrt(variance)

        self._feature_mean = mean.to(dtype=self.dtype)
        self._feature_std = std.to(dtype=self.dtype)
