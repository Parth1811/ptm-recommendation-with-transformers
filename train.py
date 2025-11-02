"""CLI entrypoint for running registered trainers."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

from beautilog import logger

from train import TRAINER_REGISTRY


def list_trainers() -> str:
    lines = ["Available trainers:"]
    for name in sorted(TRAINER_REGISTRY):
        trainer_cls = TRAINER_REGISTRY[name]
        doc = (trainer_cls.__doc__ or "").strip().splitlines()[0] if trainer_cls.__doc__ else ""
        if doc:
            lines.append(f"  - {name}: {doc}")
        else:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def resolve_trainer(name: str):
    normalised = name.strip()
    registry = {key: value for key, value in TRAINER_REGISTRY.items()}
    lower_registry = {key.lower(): value for key, value in TRAINER_REGISTRY.items()}

    trainer_cls = registry.get(normalised)
    if trainer_cls:
        return trainer_cls

    trainer_cls = lower_registry.get(normalised.lower())
    if trainer_cls:
        return trainer_cls

    # Allow requesting by dropping the trailing "Trainer" suffix.
    if normalised.lower().endswith("trainer"):
        base_name = normalised[:-7]
    else:
        base_name = normalised

    for key, value in TRAINER_REGISTRY.items():
        if key.lower().startswith(base_name.lower()):
            return value

    raise KeyError(normalised)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a trainer from the registry.")
    parser.add_argument("trainer_name", nargs="?", help="Name of the trainer to execute.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available trainers and exit.",
    )

    args = parser.parse_args(argv)

    if args.list:
        print(list_trainers())
        return 0

    if not args.trainer_name:
        parser.error("trainer_name is required when --list is not provided.")

    try:
        trainer_cls = resolve_trainer(args.trainer_name)
    except KeyError:
        logger.error("Unknown trainer '%s'.\n%s", args.trainer_name, list_trainers())
        return 1

    trainer = trainer_cls()

    start_time = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start_time
    logger.info("Training completed in %.2f seconds (%.2f minutes).", elapsed, elapsed / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
