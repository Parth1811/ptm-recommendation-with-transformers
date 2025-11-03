"""Evaluate CLIP encoder on ImageNet balanced batches and persist embeddings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from beautilog import logger

from config import ClipEvaluationConfig, ConfigParser, ImageNetDatasetConfig
from dataloader import ImageNetBalancedDataLoader, ImageNetDataset
from model import ClipImageEncoder


def _prepare_dataset_config(
    data_config: ImageNetDatasetConfig,
    eval_config: ClipEvaluationConfig,
) -> ImageNetDatasetConfig:
    """Return a copy of the dataset config with evaluation-safe defaults applied."""
    cfg = data_config
    updates: dict[str, object] = {}

    if eval_config.dataset_split and eval_config.dataset_split != data_config.split:
        updates["split"] = eval_config.dataset_split

    # Deterministic evaluation batches
    if data_config.shuffle:
        updates["shuffle"] = False
    if data_config.drop_last:
        updates["drop_last"] = False

    if eval_config.cache_directory_override is not None:
        updates["cache_dir"] = eval_config.cache_directory_override

    if not updates:
        return cfg

    return replace(cfg, **updates)


def _labels_to_names(labels: Sequence[int], class_names: Sequence[str]) -> np.ndarray:
    if not class_names:
        return np.array(labels, dtype=np.int64)
    names = []
    for label in labels:
        if 0 <= label < len(class_names):
            names.append(class_names[label])
        else:
            names.append(f"class_{label}")
    return np.asarray(names, dtype=object)


def main() -> None:
    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    dataset_config = ConfigParser.get(ImageNetDatasetConfig)
    dataset_config = _prepare_dataset_config(dataset_config, eval_config)

    clip_encoder = ClipImageEncoder(
        model_name=eval_config.model_name,
        device=eval_config.device,
        precision=eval_config.precision,  # type: ignore[arg-type]
        normalize_features=eval_config.normalize_features,
    )
    transform = clip_encoder.build_transform(train=False)

    dataset = ImageNetDataset(
        config=dataset_config,
        split=dataset_config.split,
        transform=transform,
        cache_dir=str(dataset_config.cache_dir),
    )

    dataloader = ImageNetBalancedDataLoader(
        dataset=dataset,
        config=dataset_config,
        persistent_workers=False,
        shuffle=False,
    )

    output_dir = Path(eval_config.output_directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving CLIP embeddings to %s", output_dir)

    batches_saved = 0

    for batch_index, (images, labels) in enumerate(dataloader):
        features = clip_encoder.encode(images)
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        class_names = _labels_to_names(labels_np.tolist(), dataset.class_names)

        sample_path = output_dir / f"imagenet_sample_{batch_index:05d}.npz"
        np.savez(
            sample_path,
            features=features_np,
            class_ids=labels_np,
            class_names=class_names,
        )
        logger.info("Saved %s", sample_path)
        batches_saved += 1

        if batches_saved >= max(1, eval_config.num_batches):
            break

    logger.info("Completed CLIP embedding extraction for %d batch(es).", batches_saved)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
