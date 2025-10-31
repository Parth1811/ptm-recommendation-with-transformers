"""Extractor interfaces for model parameter handling."""

from __future__ import annotations

from .base_extractor import BaseExtractor
from .detector_based_extractor import DetectorBasedExtractor
from .hf_pipeline_extractor import HuggingFacePipelineExtractor
from .siglip_extractor import SiglipExtractor

EXTRACTOR_CLASSES: dict[str, type[BaseExtractor]] = {
    "HuggingFacePipelineExtractor": HuggingFacePipelineExtractor,
    "SiglipExtractor": SiglipExtractor,
    "DetectorBasedExtractor": DetectorBasedExtractor,
}
