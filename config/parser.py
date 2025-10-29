"""Central configuration parser utilities."""

from __future__ import annotations

import ast
import configparser
import logging
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar

from .base import SubSectionParser

T = TypeVar("T", bound=SubSectionParser)


class ConfigParser:
    """Instantiate configuration subsections from a shared config file."""

    CONFIG_FILENAME = "config.ini"
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
        if cls._payload is not None:
            return

        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve case for downstream dataclasses
        with config_path.open(encoding="utf-8") as handle:
            parser.read_file(handle)

        cls._payload = cls._build_payload(parser)

    @staticmethod
    def _coerce_value(raw_value: str) -> Any:
        """Best-effort conversion from INI string values to Python primitives."""
        value = raw_value.strip()
        lower_value = value.lower()
        if lower_value in {"true", "false"}:
            return lower_value == "true"

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    @classmethod
    def _build_payload(cls, parser: configparser.ConfigParser) -> Mapping[str, Any]:
        """Convert the configparser contents to a Mapping usable by dataclasses."""
        payload: dict[str, Any] = {}

        if parser.defaults():
            payload.update(
                {key: cls._coerce_value(value) for key, value in parser.defaults().items()}
            )

        for section in parser.sections():
            options = {
                key: cls._coerce_value(value)
                for key, value in parser.items(section, raw=True)
            }
            payload[section] = options

        return payload

    @classmethod
    def get(cls, parser_type: Type[T]) -> T:
        """Return an instantiated configuration subsection parser."""
        if cls._payload is None:
            cls.load()

        if issubclass(parser_type, SubSectionParser):
            return parser_type.from_dict(cls._payload)

        raise TypeError(f"Unsupported parser type: {parser_type}")
