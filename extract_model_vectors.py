"""Example script showing how to load a Hugging Face model via the extractor."""

from __future__ import annotations

import csv
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


def main() -> None:
    ConfigParser.load()
    extractor_config = ConfigParser.get(ExtractorConfig)
    registry = extractor_config.extractor_registry
    default_class_name = extractor_config.default_extractor_class
    default_extractor_cls = EXTRACTOR_CLASSES.get(default_class_name)


    csv_path = Path("constants/models.csv")
    models = get_models(csv_path)

    for model_name, model_url in models:
        try:
            logger.info(f"Processing model: {model_name} from {model_url}")
            class_name = registry.get(model_name, default_class_name)
            extractor_cls = EXTRACTOR_CLASSES.get(class_name, default_extractor_cls)
            if class_name not in EXTRACTOR_CLASSES:
                logger.warning(
                    f"Extractor class '{class_name}' not found. "
                    f"Defaulting to {default_class_name}."
                )

            extractor = extractor_cls(model_url=model_url, name=model_name)
            compressed = extractor.extract()
            output_path = extractor.save(compressed)
            logger.info(f"Compressed parameters for {model_name} saved to {output_path}")
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
