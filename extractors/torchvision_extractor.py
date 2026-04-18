"""Extractor for torchvision CNN models (AlexNet, ResNet, GoogLeNet)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    AlexNet_Weights,
    GoogLeNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Maps (arch, source_dataset) -> torchvision pretrained weights.
# Only imagenet-sourced models are available via torchvision.
# All other source datasets require a PARC .pth file.
_WEIGHT_MAP: dict[tuple[str, str], object] = {
    ("alexnet",   "imagenet"): AlexNet_Weights.IMAGENET1K_V1,
    ("resnet18",  "imagenet"): ResNet18_Weights.IMAGENET1K_V1,
    ("resnet50",  "imagenet"): ResNet50_Weights.IMAGENET1K_V2,
    ("googlenet", "imagenet"): GoogLeNet_Weights.IMAGENET1K_V1,
}

# Number of output classes for each PARC source dataset
_NUM_CLASSES: dict[str, int] = {
    "nabird":        555,
    "oxford_pets":   37,
    "cub200":        200,
    "caltech101":    101,
    "stanford_dogs": 120,
    "voc2007":       21,
    "cifar10":       10,
    "imagenet":      1000,
}


class TorchvisionModelExtractor(BaseExtractor):
    """Extract and compress parameters from a torchvision CNN.

    For imagenet-sourced models the weights are downloaded automatically
    from torchvision. For all other source datasets a path to the PARC
    .pth checkpoint must be provided via ``pth_path``.

    The extraction pipeline (K-means clustering → fixed 8192-dim vector)
    is inherited from BaseExtractor and is identical to the pipeline used
    by HuggingFacePipelineExtractor.
    """

    def __init__(
        self,
        arch: str,
        source_dataset: str,
        pth_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            arch: torchvision model name, e.g. ``"alexnet"``, ``"resnet50"``.
            source_dataset: Dataset the model was pre-trained on, e.g. ``"imagenet"``.
            pth_path: Path to PARC .pth checkpoint. Required when
                ``(arch, source_dataset)`` is not in the imagenet weight map.
        """
        name = f"{arch}_{source_dataset}"
        super().__init__(name=name)
        self.arch = arch
        self.source_dataset = source_dataset
        self.pth_path = Path(pth_path) if pth_path is not None else None

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    def load_parameters(self) -> list[np.ndarray]:
        """Load model weights and return them as a list of column vectors."""
        model = self._load_model()
        state_dict = model.state_dict()
        parameters = [
            tensor.detach().cpu().numpy().reshape(-1, 1)
            for tensor in state_dict.values()
        ]
        if not parameters:
            raise ValueError(f"No parameters found for ({self.arch}, {self.source_dataset}).")
        return parameters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        key = (self.arch, self.source_dataset)

        if key in _WEIGHT_MAP:
            weights = _WEIGHT_MAP[key]
            logger.info("Loading %s with torchvision pretrained weights: %s", self.arch, weights)
            model = getattr(tv_models, self.arch)(weights=weights)

        elif self.pth_path is not None:
            if not self.pth_path.exists():
                raise FileNotFoundError(
                    f"PARC model checkpoint not found: {self.pth_path}\n"
                    f"Download from https://www.dropbox.com/s/gk32wdqmf19lnmt/models.zip"
                )
            num_classes = _NUM_CLASSES.get(self.source_dataset, 1000)
            kwargs: dict = {}
            if self.arch == "googlenet":
                kwargs = {"aux_logits": False, "init_weights": False}
            logger.info(
                "Loading %s from %s (num_classes=%d)",
                self.arch, self.pth_path, num_classes,
            )
            model = getattr(tv_models, self.arch)(
                weights=None, num_classes=num_classes, **kwargs
            )
            state_dict = torch.load(self.pth_path, map_location="cpu")
            # DataParallel checkpoints have 'module.' prefix
            if all(k.startswith("module.") for k in state_dict):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        else:
            raise ValueError(
                f"No torchvision pretrained weights for ({self.arch!r}, {self.source_dataset!r}) "
                f"and no pth_path was provided. Pass pth_path=<path to PARC .pth file>."
            )

        return model.eval()
