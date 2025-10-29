"""Base utilities for parsing configuration subsections."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin


class SubSectionParser:
    """Shared interface for configuration section loaders."""

    SECTION: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Any:
        """Instantiate the subclass using values from its named section."""
        section = payload.get(cls.SECTION, payload)

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to use SubSectionParser.")

        kwargs = {}
        for field in fields(cls):
            if not field.init:
                continue

            if field.name in section:
                value = section[field.name]
            elif field.default is not MISSING:
                value = field.default
            elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                value = field.default_factory()  # type: ignore[misc]
            else:
                raise KeyError(f"Missing required config value: {field.name}")

            if value is None:
                kwargs[field.name] = None
                continue

            field_type = field.type
            origin = get_origin(field_type)
            args = get_args(field_type)

            if field_type is Path or Path in args:
                if not isinstance(value, Path):
                    value = Path(value)

            kwargs[field.name] = value

        return cls(**kwargs)
