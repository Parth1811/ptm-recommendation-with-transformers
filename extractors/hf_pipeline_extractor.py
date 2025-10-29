"""Hugging Face extractor that loads models via the transformers API."""

from __future__ import annotations

import logging
from typing import Iterator
from urllib.parse import urlparse

import numpy as np
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class HuggingFacePipelineExtractor(BaseExtractor):
    """Extract model weights from a Hugging Face repository."""

    def __init__(
        self,
        model_url: str,
        name: str,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        model_id = self._normalize_model_id(model_url)
        super().__init__(name=name, **kwargs)
        self.model_id = model_id
        self.cache_dir = cache_dir

    def load_parameters(self) -> list[np.ndarray]:
        model = self._load_model()
        state_dict = model.state_dict()
        ordered_keys = list(self._ordered_state_keys(model))
        seen_keys: set[str] = set()
        sorted_tensors = []
        for key in ordered_keys:
            tensor = state_dict.get(key)
            if tensor is None:
                continue
            sorted_tensors.append(tensor)
            seen_keys.add(key)

        # verify if any parameters were not seen during ordering
        for key, tensor in state_dict.items():
            if key not in seen_keys:
                logger.warning(f"Unseen Key: {key}")
                sorted_tensors.append(tensor)

        parameters = [
            tensor.detach().cpu().numpy().reshape(-1, 1)
            for tensor in sorted_tensors
        ]

        if not parameters:
            raise ValueError(f"No parameters discovered for model '{self.model_id}'.")

        return parameters

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        if config.model_type in {"vit", "beit", "deit"}:
            return AutoModelForImageClassification.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            ).eval()
        return AutoModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).eval()

    @staticmethod
    def _normalize_model_id(model_reference: str) -> str:
        parsed = urlparse(model_reference)
        if parsed.scheme and parsed.netloc:
            return parsed.path.strip("/")
        return model_reference.strip("/")

    @staticmethod
    def _ordered_state_keys(model) -> Iterator[str]:
        """Yield parameter and buffer keys following module registration order."""
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for name, _ in module.named_parameters(recurse=False):
                yield f"{prefix}{name}"
            for name, _ in module.named_buffers(recurse=False):
                yield f"{prefix}{name}"
