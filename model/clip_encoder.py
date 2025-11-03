"""CLIP encoder utilities for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, CLIPProcessor


Precision = Literal["fp32", "fp16", "bf16"]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@dataclass
class ClipTransformParams:
    """Parameters describing the CLIP image preprocessing pipeline."""

    resize_shorter_side: int
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    image_size: int


class ClipImageEncoder(nn.Module):
    """Wrapper around an OpenAI CLIP vision encoder."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str | torch.device | None = None,
        precision: Precision = "fp32",
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.precision: Precision = precision
        self.normalize_features = normalize_features

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.precision == "fp16":
            self.model.half()
        elif self.precision == "bf16":
            self.model.bfloat16()

        image_processor = self.processor.image_processor
        size = image_processor.size
        if isinstance(size, dict):
            resize_shorter_side = int(size.get("shortest_edge", next(iter(size.values()))))
        else:
            resize_shorter_side = int(size)
        crop_size_dict = getattr(image_processor, "crop_size", {"height": image_processor.size})
        if isinstance(crop_size_dict, dict):
            crop_size = int(crop_size_dict.get("height") or crop_size_dict.get("width") or resize_shorter_side)
        else:
            crop_size = int(crop_size_dict)

        self.transform_params = ClipTransformParams(
            resize_shorter_side=resize_shorter_side,
            crop_size=crop_size,
            mean=tuple(image_processor.image_mean),
            std=tuple(image_processor.image_std),
            image_size=crop_size,
        )

    def build_transform(self, train: bool = False) -> transforms.Compose:
        """Return torchvision transforms matching the CLIP preprocessing pipeline."""
        params = self.transform_params
        common = [
            transforms.ToTensor(),
            transforms.Normalize(mean=params.mean, std=params.std),
        ]

        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(params.crop_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    *common,
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize(params.resize_shorter_side, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(params.crop_size),
                *common,
            ]
        )

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed images into CLIP feature space."""
        inputs = pixel_values.to(self.device)

        if self.precision == "fp16":
            inputs = inputs.half()
        elif self.precision == "bf16":
            inputs = inputs.bfloat16()

        features = self.model.get_image_features(pixel_values=inputs)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return features.float()
