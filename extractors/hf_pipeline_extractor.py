"""Hugging Face extractor that loads models via the transformers API."""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

from .base_extractor import BaseExtractor


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

    def load_parameters(self) -> np.ndarray:
        model = self._load_model()
        parameters = [
            tensor.detach().cpu().numpy().reshape(-1, 1)
            for tensor in model.state_dict().values()
        ]
        if not parameters:
            raise ValueError(f"No parameters discovered for model '{self.model_id}'.")
        return np.vstack(parameters)

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
