"""Detector-based extractor leveraging the Hugging Face pipeline utilities."""

from __future__ import annotations

import detectors
import timm
import torch

from .hf_pipeline_extractor import HuggingFacePipelineExtractor


class DetectorBasedExtractor(HuggingFacePipelineExtractor):
    """Load models using the detectors SDK conventions with timm backends."""

    def _load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(self.model_id, pretrained=True)
        model = model.to(device).eval()

        # Preserve the detectors-style transform so callers can reuse it.
        self.transform = detectors.create_transform(model)
        return model
