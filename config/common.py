"""Extractor-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import torch

from .base import SubSectionParser


@dataclass
class CommonConfig(SubSectionParser):
    SECTION: ClassVar[str] = "global"

    cuda_device: Optional[int] = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"