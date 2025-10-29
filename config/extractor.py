"""Extractor-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from .base import SubSectionParser


@dataclass(slots=True)
class ExtractorConfig(SubSectionParser):
    SECTION: ClassVar[str] = "extractor"

    cluster_count: int = 10
    model_output_folder: str = "extracted/models"
