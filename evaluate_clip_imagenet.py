"""Evaluate CLIP encoder across ImageNet splits and persist embeddings."""

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
    *,
    split_override: str | None = None,
) -> ImageNetDatasetConfig:
    """Return a copy of the dataset config with evaluation-safe defaults applied."""
    cfg = data_config
    updates: dict[str, object] = {}

    target_split = split_override or eval_config.dataset_split
    if target_split and target_split != data_config.split:
        updates["split"] = target_split

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


def _take_from_buffer(buffer: list[dict[str, np.ndarray]], count: int) -> list[dict[str, np.ndarray]]:
    """Remove up to `count` samples from the front of the buffer."""
    collected: list[dict[str, np.ndarray]] = []
    remaining = count

    while remaining > 0 and buffer:
        fragment = buffer[0]
        fragment_size = len(fragment["labels"])
        if fragment_size <= remaining:
            collected.append(fragment)
            buffer.pop(0)
            remaining -= fragment_size
        else:
            collected.append(
                {
                    "features": fragment["features"][:remaining],
                    "labels": fragment["labels"][:remaining],
                    "class_names": fragment["class_names"][:remaining],
                }
            )
            buffer[0] = {
                "features": fragment["features"][remaining:],
                "labels": fragment["labels"][remaining:],
                "class_names": fragment["class_names"][remaining:],
            }
            remaining = 0

    return collected


def _save_shard(
    split_dir: Path,
    split_name: str,
    shard_index: int,
    fragments: Sequence[dict[str, np.ndarray]],
) -> None:
    """Persist a chunk of encoded samples to disk."""
    features = np.concatenate([fragment["features"] for fragment in fragments], axis=0)
    class_ids = np.concatenate([fragment["labels"] for fragment in fragments], axis=0)
    class_names = np.concatenate([fragment["class_names"] for fragment in fragments], axis=0)

    shard_path = split_dir / f"{split_name}_clip_{shard_index:05d}.npz"
    np.savez(
        shard_path,
        features=features,
        class_ids=class_ids,
        class_names=class_names,
    )
    logger.info("Saved %s samples=%d", shard_path, features.shape[0])


def _flush_shards(
    buffer: list[dict[str, np.ndarray]],
    buffer_count: int,
    *,
    shard_size: int,
    split_dir: Path,
    split_name: str,
    shard_index: int,
    save_partial: bool = False,
) -> tuple[int, int]:
    """Write full shards (and optionally the tail) to disk."""
    while shard_size > 0 and buffer_count >= shard_size:
        fragments = _take_from_buffer(buffer, shard_size)
        shard_count = sum(len(fragment["labels"]) for fragment in fragments)
        buffer_count -= shard_count
        _save_shard(split_dir, split_name, shard_index, fragments)
        shard_index += 1

    if save_partial and buffer_count > 0:
        fragments = _take_from_buffer(buffer, buffer_count)
        _save_shard(split_dir, split_name, shard_index, fragments)
        shard_index += 1
        buffer_count = 0

    return shard_index, buffer_count


def _process_split(
    split_name: str,
    sample_target: int,
    *,
    clip_encoder: ClipImageEncoder,
    base_dataset_config: ImageNetDatasetConfig,
    eval_config: ClipEvaluationConfig,
    output_dir: Path,
) -> None:
    """Encode and persist embeddings for a single dataset split."""
    if sample_target <= 0:
        logger.info("Skipping split '%s' because requested sample count is <= 0.", split_name)
        return

    dataset_config = _prepare_dataset_config(base_dataset_config, eval_config, split_override=split_name)
    transform = clip_encoder.build_transform(train=dataset_config.split.lower() == "train")

    try:
        dataset = ImageNetDataset(
            config=dataset_config,
            split=dataset_config.split,
            transform=transform,
            cache_dir=str(dataset_config.cache_dir),
        )
    except Exception as exc:  # pragma: no cover - surface dataset load issues to callers
        raise RuntimeError(f"Failed to load ImageNet split '{split_name}': {exc}") from exc

    dataloader = ImageNetBalancedDataLoader(
        dataset=dataset,
        config=dataset_config,
        persistent_workers=False,
        shuffle=False,
    )

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Processing split '%s' into %s (target=%d samples, shard_size=%d)",
        split_name,
        split_dir,
        sample_target,
        eval_config.samples_per_shard,
    )

    shard_size = eval_config.samples_per_shard if eval_config.samples_per_shard > 0 else sample_target
    buffer: list[dict[str, np.ndarray]] = []
    buffer_count = 0
    shard_index = 0
    samples_processed = 0

    for images, labels in dataloader:
        if samples_processed >= sample_target:
            break

        remaining = sample_target - samples_processed
        if remaining <= 0:
            break

        if labels.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        features = clip_encoder.encode(images)
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        class_names_np = _labels_to_names(labels_np.tolist(), dataset.class_names)

        buffer.append(
            {"features": features_np, "labels": labels_np, "class_names": class_names_np}
        )
        batch_size = labels_np.shape[0]
        buffer_count += batch_size
        samples_processed += batch_size

        shard_index, buffer_count = _flush_shards(
            buffer,
            buffer_count,
            shard_size=shard_size,
            split_dir=split_dir,
            split_name=split_name,
            shard_index=shard_index,
        )

    shard_index, buffer_count = _flush_shards(
        buffer,
        buffer_count,
        shard_size=shard_size,
        split_dir=split_dir,
        split_name=split_name,
        shard_index=shard_index,
        save_partial=True,
    )

    if samples_processed < sample_target:
        logger.warning(
            "Requested %d samples for split '%s' but only processed %d.",
            sample_target,
            split_name,
            samples_processed,
        )

    logger.info(
        "Completed split '%s' with %d samples across %d shard(s).",
        split_name,
        samples_processed,
        shard_index,
    )


def main() -> None:
    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    dataset_config = ConfigParser.get(ImageNetDatasetConfig)

    clip_encoder = ClipImageEncoder(
        model_name=eval_config.model_name,
        device=eval_config.device,
        precision=eval_config.precision,  # type: ignore[arg-type]
        normalize_features=eval_config.normalize_features,
    )

    output_dir = Path(eval_config.output_directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving CLIP embeddings to %s", output_dir)

    if not eval_config.split_sample_counts:
        raise ValueError("No splits configured in 'split_sample_counts'.")

    for split_name, sample_target in eval_config.split_sample_counts.items():
        _process_split(
            split_name,
            sample_target,
            clip_encoder=clip_encoder,
            base_dataset_config=dataset_config,
            eval_config=eval_config,
            output_dir=output_dir,
        )

    logger.info("Completed CLIP embedding extraction for all configured splits.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
