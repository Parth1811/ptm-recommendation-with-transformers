"""Extractor-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from .base import SubSectionParser


@dataclass
class ExtractorConfig(SubSectionParser):
    SECTION: ClassVar[str] = "extractor"

    output_stack_size: int
    model_output_folder: str
    default_extractor_class: str = "HuggingFacePipelineExtractor"
    extractor_registry: dict[str, str] = field(default_factory=dict)
