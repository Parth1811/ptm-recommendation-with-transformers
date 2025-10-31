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
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Parameter directory not found: {self.root_dir}")

        self.dtype = dtype
        self.flatten = flatten
        extensions = tuple(file_extensions) if file_extensions else (".npz",)
        self.files = sorted(
            path
            for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix in extensions
        )
        if not self.files:
            raise ValueError(f"No parameter archives found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.files[index]
        with np.load(file_path) as archive:
            parameters = archive["parameters"]

        if self.flatten:
            parameters = parameters.reshape(-1)

        tensor = torch.as_tensor(parameters, dtype=self.dtype)
        return tensor
