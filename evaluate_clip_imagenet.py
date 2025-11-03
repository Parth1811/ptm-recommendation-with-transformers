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


def _save_shard(
    split_dir: Path,
    split_name: str,
    shard_index: int,
    features: np.ndarray,
    class_ids: np.ndarray,
    class_names: np.ndarray,
    *,
    actual_batches: int,
    target_batches: int,
) -> None:
    """Persist a chunk of encoded samples to disk."""
    shard_path = split_dir / f"{split_name}_clip_{shard_index:05d}.npz"
    np.savez(
        shard_path,
        features=features,
        class_ids=class_ids,
        class_names=class_names,
        actual_batches=np.array([actual_batches], dtype=np.int32),
        target_batches=np.array([target_batches], dtype=np.int32),
    )
    logger.info(
        "Saved %s shape=%s actual_batches=%d target_batches=%d",
        shard_path,
        tuple(features.shape),
        actual_batches,
        target_batches,
    )


def _emit_shard(
    buffer: list[dict[str, np.ndarray]],
    *,
    shard_size: int,
    split_dir: Path,
    split_name: str,
    shard_index: int,
    pad_to_full: bool,
) -> int:
    """Write the next shard worth of balanced batches to disk."""
    if not buffer:
        return shard_index

    emit_count = min(shard_size, len(buffer))
    shard_entries = buffer[:emit_count]
    del buffer[:emit_count]

    features = np.stack([entry["features"] for entry in shard_entries], axis=0)
    class_ids = np.stack([entry["class_ids"] for entry in shard_entries], axis=0)
    class_names = np.stack([entry["class_names"] for entry in shard_entries], axis=0)

    actual_batches = features.shape[0]
    target_batches = shard_size

    if pad_to_full and actual_batches < shard_size:
        num_classes = features.shape[1]
        embed_dim = features.shape[2]
        pad_batches = shard_size - actual_batches
        features = np.concatenate(
            [
                features,
                np.zeros((pad_batches, num_classes, embed_dim), dtype=features.dtype),
            ],
            axis=0,
        )
        class_ids = np.concatenate(
            [
                class_ids,
                np.full((pad_batches, num_classes), -1, dtype=class_ids.dtype),
            ],
            axis=0,
        )
        class_names = np.concatenate(
            [
                class_names,
                np.full((pad_batches, num_classes), "", dtype=object),
            ],
            axis=0,
        )

    _save_shard(
        split_dir,
        split_name,
        shard_index,
        features,
        class_ids,
        class_names,
        actual_batches=actual_batches,
        target_batches=target_batches,
    )
    return shard_index + 1


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

    class_count = dataloader.num_classes
    if class_count <= 0:
        raise ValueError(f"Dataset split '{split_name}' reported zero classes.")

    batches_target = (sample_target + class_count - 1) // class_count
    if batches_target == 0:
        logger.info("No batches requested for split '%s'; skipping.", split_name)
        return

    if sample_target % class_count != 0:
        logger.warning(
            "Requested %d samples for split '%s' which is not divisible by %d classes; "
            "processing %d batches (%d samples) instead.",
            sample_target,
            split_name,
            class_count,
            batches_target,
            batches_target * class_count,
        )

    configured_shard_size = eval_config.batches_per_shard
    shard_size = configured_shard_size if configured_shard_size and configured_shard_size > 0 else batches_target
    pad_to_full = bool(eval_config.pad_to_full_shard)

    logger.info(
        "Processing split '%s' into %s (target_batches=%d, classes=%d, shard_batches=%d)",
        split_name,
        split_dir,
        batches_target,
        class_count,
        shard_size,
    )

    batch_buffer: list[dict[str, np.ndarray]] = []
    shard_index = 0
    batches_processed = 0

    for images, labels in dataloader:
        if batches_processed >= batches_target:
            break

        features = clip_encoder.encode(images)
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        class_names_np = _labels_to_names(labels_np.tolist(), dataset.class_names)

        batch_buffer.append(
            {"features": features_np, "class_ids": labels_np, "class_names": class_names_np}
        )
        batches_processed += 1

        if shard_size > 0 and len(batch_buffer) >= shard_size:
            shard_index = _emit_shard(
                batch_buffer,
                shard_size=shard_size,
                split_dir=split_dir,
                split_name=split_name,
                shard_index=shard_index,
                pad_to_full=False,
            )

    if batches_processed < batches_target:
        logger.warning(
            "Requested %d batches for split '%s' but only processed %d; requested samples=%d actual samples=%d.",
            batches_target,
            split_name,
            batches_processed,
            sample_target,
            batches_processed * class_count,
        )

    if batch_buffer:
        if shard_size > 0 and pad_to_full:
            final_shard_size = shard_size
            final_pad = True
        else:
            final_shard_size = len(batch_buffer)
            final_pad = False

        shard_index = _emit_shard(
            batch_buffer,
            shard_size=final_shard_size,
            split_dir=split_dir,
            split_name=split_name,
            shard_index=shard_index,
            pad_to_full=final_pad,
        )

    logger.info(
        "Completed split '%s' with %d samples across %d shard(s).",
        split_name,
        batches_processed * class_count,
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
