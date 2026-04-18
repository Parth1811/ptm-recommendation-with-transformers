"""Extract CLIP embeddings for datasets missing from the server.

Reads the existing extracted datasets directory, compares against
the full list needed for training + benchmark evaluation, and runs
CLIP extraction only for the missing ones.

Usage:
    python extract_missing_datasets.py                  # Extract all missing
    python extract_missing_datasets.py --dry-run        # Just show what's missing
    python extract_missing_datasets.py --only Aircraft Cars  # Extract specific ones

Existing datasets on server (already extracted):
    caltech_101, cifar_10, cifar_100, deepfashion_inshop, fashion_mnist,
    fashion_product, ham10000, imagenet_1k, mnist, svhn, usps_digits

Missing datasets needed for training pipeline + benchmark:
    Aircraft, Cars, SUN397, DTD, Pets, pokemon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from beautilog import logger

from config import (
    ClipEvaluationConfig,
    ConfigParser,
    DatasetLoaderDefaultsConfig,
    DatasetRegistryConfig,
)


def _make_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


# All datasets needed by the training pipeline and benchmark evaluation.
# Maps display name -> config.ini registry key
REQUIRED_DATASETS = {
    # Already in config.ini [dataset_loader] registry:
    "MNIST": "MNIST",
    "SVHN": "SVHN",
    "USPS Digits": "USPS Digits",
    "Fashion-MNIST": "Fashion-MNIST",
    "Deepfashion Inshop": "Deepfashion Inshop",
    "Fashion Product": "Fashion Product",
    "ImageNet 1k": "ImageNet 1k",
    "CIFAR-10": "CIFAR-10",
    "CIFAR-100": "CIFAR-100",
    "Caltech-101": "Caltech-101",
    "HAM10000": "HAM10000",
    "Aircraft": "Aircraft",
    "SUN397": "SUN397",
    "Cars": "Cars",
    "DTD": "DTD",
    "Pets": "Pets",
    "pokemon": "pokemon",
}


def find_missing_datasets(output_dir: Path) -> list[str]:
    """Compare required datasets against what's already extracted.

    Returns list of dataset names that need extraction.
    """
    existing_slugs = set()
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir():
                existing_slugs.add(d.name)

    missing = []
    for display_name in REQUIRED_DATASETS:
        slug = _make_slug(display_name)
        if slug not in existing_slugs:
            missing.append(display_name)

    return missing


def extract_datasets(dataset_names: list[str]) -> None:
    """Run CLIP extraction for the specified datasets.

    Reuses the same extraction pipeline as evaluate_clip_imagenet.py.
    """
    from evaluate_clip_imagenet import _process_dataset

    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    defaults = ConfigParser.get(DatasetLoaderDefaultsConfig)
    registry = ConfigParser.get(DatasetRegistryConfig)

    from model import ClipImageEncoder

    clip_encoder = ClipImageEncoder(
        model_name=eval_config.model_name,
        device=eval_config.device,
        precision=eval_config.precision,
        normalize_features=eval_config.normalize_features,
    )

    configured_datasets = registry.loader_registry
    if not configured_datasets:
        logger.error("Dataset registry is empty in config.ini")
        sys.exit(1)

    for dataset_name in dataset_names:
        registry_key = REQUIRED_DATASETS.get(dataset_name, dataset_name)
        dataset_spec = configured_datasets.get(registry_key)

        if dataset_spec is None:
            logger.error(
                "Dataset '%s' (key='%s') not found in config.ini [dataset_loader] registry. "
                "Add it to config.ini first.",
                dataset_name,
                registry_key,
            )
            continue

        logger.info("=" * 60)
        logger.info("Extracting: %s", dataset_name)
        logger.info("=" * 60)

        try:
            _process_dataset(
                dataset_name,
                dataset_spec,
                clip_encoder=clip_encoder,
                defaults=defaults,
                eval_config=eval_config,
            )
            logger.info("Completed: %s", dataset_name)
        except Exception as exc:
            logger.error("Failed to extract '%s': %s", dataset_name, exc)
            import traceback
            logger.error(traceback.format_exc())
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for missing datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what's missing, don't extract",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Extract only these specific datasets (by display name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: from config.ini [clip_evaluation] output_directory)",
    )
    args = parser.parse_args()

    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    output_dir = args.output_dir or Path(eval_config.output_directory).expanduser()

    logger.info("Output directory: %s", output_dir)
    logger.info("Scanning for existing extracted datasets...")

    missing = find_missing_datasets(output_dir)

    if args.only:
        # Filter to only requested datasets
        requested = set(args.only)
        missing = [d for d in missing if d in requested]
        # Also include explicitly requested ones even if they exist (re-extract)
        for name in args.only:
            if name not in missing and name in REQUIRED_DATASETS:
                missing.append(name)

    if not missing:
        logger.info("All required datasets are already extracted!")
        return

    existing_count = len(REQUIRED_DATASETS) - len(find_missing_datasets(output_dir))
    logger.info(
        "Dataset status: %d/%d extracted, %d missing",
        existing_count,
        len(REQUIRED_DATASETS),
        len(missing),
    )
    logger.info("")

    for i, name in enumerate(missing, 1):
        slug = _make_slug(name)
        hf_id = "—"
        registry = ConfigParser.get(DatasetRegistryConfig).loader_registry
        spec = registry.get(REQUIRED_DATASETS.get(name, name), {})
        if isinstance(spec, dict):
            hf_id = spec.get("dataset_name", "—")
        logger.info("  [%d] %s (slug=%s, hf=%s)", i, name, slug, hf_id)

    logger.info("")

    if args.dry_run:
        logger.info("Dry run — no extraction performed.")
        return

    logger.info("Starting extraction for %d dataset(s)...", len(missing))
    torch.set_grad_enabled(False)
    extract_datasets(missing)
    logger.info("Done! All missing datasets extracted.")


if __name__ == "__main__":
    main()
