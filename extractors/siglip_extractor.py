"""Extractor for SigLIP image classification models hosted on Hugging Face."""

from __future__ import annotations

from transformers import AutoImageProcessor, SiglipForImageClassification

from .hf_pipeline_extractor import HuggingFacePipelineExtractor


class SiglipExtractor(HuggingFacePipelineExtractor):
    """Loads a SigLIP image classifier for parameter extraction."""

    def _load_model(self):
        # Store the processor so downstream code can reuse the same preprocessing settings.
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        return SiglipForImageClassification.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        ).eval()
