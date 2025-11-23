"""Script to pre-download HuggingFace models from a CSV file."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from beautilog import logger
from huggingface_hub import snapshot_download


def get_models(csv_path: Path) -> list[tuple[str, str]]:
    """Read models from CSV file."""
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


def download_model(model_name: str, model_url: str, cache_dir: str | None = None) -> None:
    """Download a single model from HuggingFace.

    Args:
        model_name: Name of the model
        model_url: HuggingFace URL
        cache_dir: Optional custom cache directory. If None, uses HuggingFace default or HF_HOME env var
    """
    try:
        # Extract repo_id from URL
        # URL format: https://huggingface.co/timm/model_name
        if "huggingface.co/" in model_url:
            repo_id = model_url.split("huggingface.co/")[-1]
        else:
            logger.error(f"Invalid HuggingFace URL: {model_url}")
            return

        logger.info(f"Downloading {model_name} from {repo_id}...")
        if cache_dir:
            logger.info(f"  Cache directory: {cache_dir}")

        # Download the model to cache
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.safetensors", "*.bin", "config.json", "*.json"],
            ignore_patterns=["*.md", "*.txt", "*.onnx", "*.msgpack"],
            cache_dir=cache_dir,
        )

        logger.info(f"✓ Successfully downloaded {model_name}")

    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")


def main() -> None:
    """Download all models from the CSV file."""
    csv_path = Path("constants/models.csv")
    # csv_path = Path("constants/models-test.csv")

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    # Check for custom cache directory from environment variable
    # Set HF_HOME environment variable to use a custom cache directory
    # Example: export HF_HOME=/path/to/your/custom/cache
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")

    if cache_dir:
        logger.info(f"Using custom HuggingFace cache directory: {cache_dir}")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Using default HuggingFace cache directory (~/.cache/huggingface)")

    models = get_models(csv_path)
    logger.info(f"Found {len(models)} models to download from {csv_path}")

    for idx, (model_name, model_url) in enumerate(models, 1):
        logger.info(f"[{idx}/{len(models)}] Processing {model_name}")
        download_model(model_name, model_url, cache_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"Download complete! All models are cached locally.")
    if cache_dir:
        logger.info(f"Cache location: {cache_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
