"""Example script showing how to load a Hugging Face model via the extractor."""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import traceback
from pathlib import Path

from beautilog import logger

from config import ConfigParser, ExtractorConfig
from extractors import EXTRACTOR_CLASSES


def get_models(csv_path: Path) -> list[tuple[str, str]]:
    models = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            name = row.get("name")
            url = row.get("url")
            if name and url:
                models.append((name, url))

    return models


def process_single_model(args: tuple[str, str, dict, str]) -> tuple[str, bool, str | None]:
    """
    Process a single model extraction.

    Args:
        args: Tuple of (model_name, model_url, registry, default_class_name)

    Returns:
        Tuple of (model_name, success, output_path or error_message)
    """
    model_name, model_url, registry, default_class_name = args

    try:
        class_name = registry.get(model_name, default_class_name)
        extractor_cls = EXTRACTOR_CLASSES.get(class_name)

        if extractor_cls is None:
            default_extractor_cls = EXTRACTOR_CLASSES.get(default_class_name)
            extractor_cls = default_extractor_cls
            logger.warning(
                f"[{model_name}] Extractor class '{class_name}' not found. "
                f"Defaulting to {default_class_name}."
            )

        extractor = extractor_cls(model_url=model_url, name=model_name)
        compressed = extractor.extract()
        output_path = extractor.save(compressed)

        return (model_name, True, str(output_path))

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (model_name, False, error_msg)


def main() -> None:
    extractor_config = ConfigParser.get(ExtractorConfig)
    registry = extractor_config.extractor_registry
    default_class_name = extractor_config.default_extractor_class

    csv_path = Path("constants/models.csv")
    # csv_path = Path("constants/models-test.csv")
    models = get_models(csv_path)

    # Determine number of workers (use all available cores)
    num_workers = min(len(models), 1)  # Limit to 100 workers to avoid overload

    print(f"Found {len(models)} models to process from {csv_path}")
    print(f"Using {num_workers} parallel workers")
    logger.info(f"Found {len(models)} models to process from {csv_path}")
    logger.info(f"Using {num_workers} parallel workers")

    # Prepare arguments for parallel processing
    task_args = [
        (model_name, model_url, registry, default_class_name)
        for model_name, model_url in models
    ]

    # Process models in parallel
    success_count = 0
    failure_count = 0

    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_single_model, task_args)

        for model_name, success, result in results:
            if success:
                success_count += 1
                logger.info(f"[{success_count}/{len(models)}] ✓ {model_name} -> {result}")
            else:
                failure_count += 1
                logger.error(f"[{failure_count} failures] ✗ {model_name}: {result}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Success: {success_count}/{len(models)}")
    logger.info(f"  Failures: {failure_count}/{len(models)}")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
