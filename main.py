"""Example script showing how to load a Hugging Face model via the extractor."""

from __future__ import annotations

import csv
from pathlib import Path

from beautilog import logger

from config import ConfigParser
from extractors import HuggingFacePipelineExtractor


def get_first_model(csv_path: Path) -> tuple[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            name = row.get("name")
            url = row.get("url")
            if name and url:
                return name, url
    raise ValueError(f"No valid model entries found in {csv_path}")


def main() -> None:
    ConfigParser.load()

    csv_path = Path("constants/models.csv")
    model_name, model_url = get_first_model(csv_path)

    extractor = HuggingFacePipelineExtractor(model_url=model_url, name=model_name)
    compressed = extractor.extract()
    output_path = extractor.save(compressed)
    logger.info(f"Compressed parameters for {model_name} saved to {output_path}")


if __name__ == "__main__":
    main()
