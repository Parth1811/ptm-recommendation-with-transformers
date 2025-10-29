"""Central configuration parser utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar

from .base import SubSectionParser
from .extractor import ExtractorConfig

T = TypeVar("T", bound=SubSectionParser)


class ConfigParser:
    """Instantiate configuration subsections from a shared config file."""

    CONFIG_FILENAME = "config.json"
    _payload: Mapping[str, Any] | None = None


    @classmethod
    def load(cls, config_path: str | Path | None = None) -> None:
        """Load the configuration file from the given path or default location."""
        config_path = (
            Path(config_path)
            if config_path
            else Path.cwd() / cls.CONFIG_FILENAME
        )
        logging.info(f"Loading configuration from {config_path}")
        if cls._payload is None:
            cls._payload = json.loads(config_path.read_text())

    @classmethod
    def get(cls, parser_type: Type[T]) -> T:
        """Return an instantiated configuration subsection parser."""

        if issubclass(parser_type, SubSectionParser):
            return parser_type.from_dict(cls._payload)

        raise TypeError(f"Unsupported parser type: {parser_type}")
