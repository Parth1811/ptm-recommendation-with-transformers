"""Evaluate CLIP encoders across multiple datasets and persist embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import torch
from beautilog import logger
from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets, load_dataset, ClassLabel

from config import (
    ClipEvaluationConfig,
    ConfigParser,
    DatasetLoaderDefaultsConfig,
    DatasetRegistryConfig,
)
from dataloader import GenericBalancedDataLoader, GenericImageDataset
from model import ClipImageEncoder


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


def _make_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _stratify_kwargs(label_column: str | None, *, dataset: HFDataset | None = None) -> dict[str, str]:
    if not label_column or dataset is None:
        return {}

    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return {"stratify_by_column": label_column}
    return {}


def _auto_split_dataset(
    dataset: HFDataset,
    *,
    ratios: Sequence[float],
    seed: int | None,
    label_column: str | None,
) -> dict[str, HFDataset]:
    if len(ratios) != 3:
        raise ValueError("Expected three ratios for train/validation/test.")
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    stratify = _stratify_kwargs(label_column, dataset=dataset)
    first_split = dataset.train_test_split(test_size=test_ratio, seed=seed, **stratify)
    train_candidate = first_split["train"]
    test_dataset = first_split["test"]

    if val_ratio <= 0:
        validation_dataset = train_candidate.select(range(0))
        train_dataset = train_candidate
    else:
        val_fraction = val_ratio / (train_ratio + val_ratio)
        stratify_second = _stratify_kwargs(label_column, dataset=train_candidate)
        second_split = train_candidate.train_test_split(test_size=val_fraction, seed=seed, **stratify_second)
        train_dataset = second_split["train"]
        validation_dataset = second_split["test"]

    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }


def _resolve_dataset_id(dataset_spec: Mapping[str, Any]) -> str:
    dataset_id = dataset_spec.get("dataset_name") or dataset_spec.get("name")
    if not dataset_id:
        raise KeyError("Dataset registry entry is missing 'dataset_name'.")
    return str(dataset_id)


def _load_dataset_splits(
    dataset_id: str,
    *,
    cache_dir: Path,
    split_definitions: Mapping[str, str] | None,
    split_aliases: Mapping[str, str],
    split_ratios: Sequence[float],
    seed: int | None,
    label_column: str | None,
    create_missing: bool,
    load_kwargs: dict[str, Any],
) -> dict[str, HFDataset]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_splits = ("train", "validation", "test")

    if split_definitions:
        logger.info("Using explicit split definitions for %s", dataset_id)
        return {
            split: load_dataset(dataset_id, split=definition, cache_dir=str(cache_dir), **load_kwargs)
            for split, definition in split_definitions.items()
        }

    dataset_obj = load_dataset(dataset_id, cache_dir=str(cache_dir), **load_kwargs)
    if isinstance(dataset_obj, DatasetDict):
        dataset_dict = dataset_obj
    else:
        dataset_dict = DatasetDict({"train": dataset_obj})

    resolved: dict[str, HFDataset] = {}
    for canonical in target_splits:
        if canonical in dataset_dict:
            resolved[canonical] = dataset_dict[canonical]
            continue
        alias = split_aliases.get(canonical)
        if alias and alias in dataset_dict:
            resolved[canonical] = dataset_dict[alias]
            continue
        for candidate in dataset_dict.keys():
            if candidate.lower() == canonical:
                resolved[canonical] = dataset_dict[candidate]
                break

    missing = [split for split in target_splits if split not in resolved]
    if missing and create_missing:
        logger.info(
            "Creating missing splits %s for %s using ratios %s.",
            missing,
            dataset_id,
            split_ratios,
        )
        combined = concatenate_datasets(list(dataset_dict.values()))
        generated = _auto_split_dataset(
            combined,
            ratios=split_ratios,
            seed=seed,
            label_column=label_column,
        )
        for split, hf_split in generated.items():
            resolved.setdefault(split, hf_split)

    return resolved


def _canonical_split_name(name: str) -> str:
    lookup = {
        "train": "train",
        "training": "train",
        "test": "test",
        "testing": "test",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
    }
    return lookup.get(name.lower(), name.lower())


def _resolve_max_samples(
    split_name: str,
    dataset_spec: Mapping[str, Any],
    defaults: DatasetLoaderDefaultsConfig,
) -> int | None:
    canonical = _canonical_split_name(split_name)

    per_split = dataset_spec.get("max_samples")
    if isinstance(per_split, Mapping):
        candidate = per_split.get(canonical)
        if candidate is None and canonical == "validation":
            candidate = per_split.get("val")
        if candidate is not None:
            return int(candidate)

    explicit_keys = [f"max_samples_{canonical}"]
    if canonical == "validation":
        explicit_keys.extend(["max_samples_val", "max_samples_valid"])
    elif canonical == "test":
        explicit_keys.append("max_samples_testing")
    elif canonical == "train":
        explicit_keys.append("max_samples_training")

    for key in explicit_keys:
        value = dataset_spec.get(key)
        if value is not None:
            return int(value)

    default_attr = f"max_samples_{canonical}"
    return getattr(defaults, default_attr, None)


def _limit_dataset_rows(
    dataset_split: HFDataset,
    *,
    max_samples: int | None,
    seed: int | None,
) -> HFDataset:
    if max_samples is None or max_samples <= 0:
        return dataset_split

    available = len(dataset_split)
    if available <= max_samples:
        return dataset_split

    shuffle_seed = seed if seed is not None else 0
    shuffled = dataset_split.shuffle(seed=shuffle_seed)
    return shuffled.select(range(max_samples))


def _extract_label_from_filename(
    value: str,
    *,
    separator: str,
    index: int,
    strip_extension: bool,
    lowercase: bool,
) -> str:
    token = value
    if separator:
        parts = value.split(separator)
        if not parts:
            token = value
        else:
            try:
                token = parts[index]
            except IndexError:
                token = parts[0]
    if strip_extension:
        token = token.rsplit(".", maxsplit=1)[0]
    if lowercase:
        token = token.lower()
    return token


def _derive_labels_from_filename(
    dataset_split: HFDataset,
    *,
    source_column: str,
    target_column: str,
    separator: str,
    index: int,
    strip_extension: bool,
    lowercase: bool,
) -> HFDataset:
    if source_column not in dataset_split.column_names:
        raise KeyError(f"Column '{source_column}' not present in dataset; cannot derive labels from filename.")

    def mapper(batch: Mapping[str, Any]) -> dict[str, Any]:
        filenames = batch[source_column]
        labels = [
            _extract_label_from_filename(
                str(filename),
                separator=separator,
                index=index,
                strip_extension=strip_extension,
                lowercase=lowercase,
            )
            for filename in filenames
        ]
        new_batch = dict(batch)
        new_batch[target_column] = labels
        return new_batch

    return dataset_split.map(
        mapper,
        batched=True,
        batch_size=1000,
        desc=f"Deriving labels from '{source_column}'",
        load_from_cache_file=False,
    )


def _save_shard(
    shard_path: Path,
    features: np.ndarray,
    class_ids: np.ndarray,
    class_names: np.ndarray,
    *,
    actual_batches: int,
    target_batches: int,
) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
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
    shard_dir: Path,
    split_name: str,
    shard_index: int,
    pad_to_full: bool,
) -> int:
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
            [features, np.zeros((pad_batches, num_classes, embed_dim), dtype=features.dtype)],
            axis=0,
        )
        class_ids = np.concatenate(
            [class_ids, np.full((pad_batches, num_classes), -1, dtype=class_ids.dtype)],
            axis=0,
        )
        class_names = np.concatenate(
            [class_names, np.full((pad_batches, num_classes), "", dtype=object)],
            axis=0,
        )

    shard_path = shard_dir / f"{split_name}_clip_{shard_index:05d}.npz"
    _save_shard(shard_path, features, class_ids, class_names, actual_batches=actual_batches, target_batches=target_batches)
    return shard_index + 1


def _process_split(
    dataset_name: str,
    split_name: str,
    hf_split: HFDataset,
    *,
    clip_encoder: ClipImageEncoder,
    defaults: DatasetLoaderDefaultsConfig,
    eval_config: ClipEvaluationConfig,
    dataset_spec: Mapping[str, Any],
    output_root: Path,
) -> None:
    transform = clip_encoder.build_transform(train=split_name.lower() == "train")
    image_column = dataset_spec.get("image_column") or defaults.image_column
    label_column = dataset_spec.get("label_column") or defaults.label_column

    torch_dataset = GenericImageDataset(
        hf_split,
        transform=transform,
        image_column=str(image_column) if image_column else None,
        label_column=str(label_column) if label_column else None,
    )

    if torch_dataset.num_classes <= 0:
        raise ValueError(f"Dataset '{dataset_name}' split '{split_name}' has no classes.")

    dataloader = GenericBalancedDataLoader(
        torch_dataset,
        drop_last=defaults.drop_last,
        shuffle=defaults.shuffle,
        seed=defaults.seed,
        num_workers=defaults.num_workers,
        pin_memory=defaults.pin_memory,
        persistent_workers=defaults.persistent_workers,
        prefetch_factor=defaults.prefetch_factor,
    )

    class_count = dataloader.num_classes
    batches_target = math.ceil(len(torch_dataset) / class_count)
    if eval_config.limit_batches_per_split is not None:
        batches_target = min(batches_target, int(eval_config.limit_batches_per_split))

    shard_size = eval_config.batches_per_shard if eval_config.batches_per_shard > 0 else batches_target
    pad_to_full = bool(eval_config.pad_to_full_shard)

    dataset_dir = output_root / _make_slug(dataset_name)
    split_dir = dataset_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Processing %s/%s -> %s (batches=%d, shard_batches=%d, classes=%d)",
        dataset_name,
        split_name,
        split_dir,
        batches_target,
        shard_size,
        class_count,
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
        class_names_np = _labels_to_names(labels_np.tolist(), dataloader.class_names)

        batch_buffer.append(
            {"features": features_np, "class_ids": labels_np, "class_names": class_names_np}
        )
        batches_processed += 1

        if shard_size > 0 and len(batch_buffer) >= shard_size:
            shard_index = _emit_shard(
                batch_buffer,
                shard_size=shard_size,
                shard_dir=split_dir,
                split_name=split_name,
                shard_index=shard_index,
                pad_to_full=False,
            )

    if batches_processed < batches_target:
        logger.warning(
            "Stopped early for %s/%s: processed %d of %d batches.",
            dataset_name,
            split_name,
            batches_processed,
            batches_target,
        )

    if batch_buffer:
        final_shard_size = shard_size if shard_size > 0 and pad_to_full else len(batch_buffer)
        final_pad = shard_size > 0 and pad_to_full
        shard_index = _emit_shard(
            batch_buffer,
            shard_size=final_shard_size,
            shard_dir=split_dir,
            split_name=split_name,
            shard_index=shard_index,
            pad_to_full=final_pad,
        )

    logger.info(
        "Completed %s/%s with %d batches -> %d shard(s).",
        dataset_name,
        split_name,
        batches_processed,
        shard_index,
    )


def _process_dataset(
    dataset_name: str,
    dataset_spec: Mapping[str, Any],
    *,
    clip_encoder: ClipImageEncoder,
    defaults: DatasetLoaderDefaultsConfig,
    eval_config: ClipEvaluationConfig,
) -> None:
    dataset_id = _resolve_dataset_id(dataset_spec)
    cache_override = eval_config.cache_directory_override
    base_cache_dir = Path(cache_override) if cache_override else defaults.cache_dir.expanduser()
    dataset_cache_dir = (base_cache_dir / _make_slug(dataset_name)).expanduser()

    split_definitions = dataset_spec.get("split_definitions")
    split_aliases = dataset_spec.get("split_aliases", {})
    split_ratios = dataset_spec.get("split_ratios", defaults.split_ratios)
    seed = dataset_spec.get("seed", defaults.seed)
    label_column = dataset_spec.get("label_column") or defaults.label_column
    create_missing = bool(dataset_spec.get("create_missing_splits", True))
    dataset_config_name = (
        dataset_spec.get("dataset_config")
        or dataset_spec.get("dataset_config_name")
        or dataset_spec.get("config_name")
    )

    load_kwargs = dict(eval_config.extra_load_kwargs)
    load_kwargs.update(dataset_spec.get("load_kwargs", {}))
    if dataset_config_name and "name" not in load_kwargs:
        load_kwargs["name"] = str(dataset_config_name)

    try:
        dataset_splits = _load_dataset_splits(
            dataset_id,
            cache_dir=dataset_cache_dir,
            split_definitions=split_definitions if isinstance(split_definitions, Mapping) else None,
            split_aliases=split_aliases if isinstance(split_aliases, Mapping) else {},
            split_ratios=split_ratios,
            seed=seed,
            label_column=str(label_column) if label_column else None,
            create_missing=create_missing,
            load_kwargs=load_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}' ({dataset_id}): {exc}") from exc

    label_from_filename = dataset_spec.get("label_from_filename")
    if label_from_filename:
        params: dict[str, Any] = {}
        if isinstance(label_from_filename, Mapping):
            params.update(label_from_filename)

        source_column = str(params.get("source_column", params.get("filename_column", "filename")))
        target_column = str(params.get("target_column", params.get("label_column", "label")))
        separator = str(params.get("separator", params.get("delimiter", "/")))
        index = int(params.get("index", params.get("label_index", 0)))
        strip_extension = bool(params.get("strip_extension", True))
        lowercase = bool(params.get("lowercase", True))

        transformed_splits: dict[str, HFDataset] = {}
        for split_name, hf_split in dataset_splits.items():
            transformed_splits[split_name] = _derive_labels_from_filename(
                hf_split,
                source_column=source_column,
                target_column=target_column,
                separator=separator,
                index=index,
                strip_extension=strip_extension,
                lowercase=lowercase,
            )
        dataset_splits = transformed_splits
        dataset_spec.setdefault("label_column", target_column)

    limit_seed: int | None = seed if isinstance(seed, int) else None
    if limit_seed is None and isinstance(defaults.seed, int):
        limit_seed = defaults.seed

    limited_splits: dict[str, HFDataset] = {}
    for split_name, hf_split in dataset_splits.items():
        max_samples = _resolve_max_samples(split_name, dataset_spec, defaults)
        limited_split = _limit_dataset_rows(
            hf_split,
            max_samples=max_samples,
            seed=limit_seed,
        )
        if len(limited_split) != len(hf_split):
            logger.info(
                "Limiting %s/%s from %d to %d samples (max=%d)",
                dataset_name,
                split_name,
                len(hf_split),
                len(limited_split),
                int(max_samples) if max_samples is not None else -1,
            )
        limited_splits[split_name] = limited_split
    dataset_splits = limited_splits

    if not dataset_splits:
        logger.warning("Dataset '%s' produced no splits; skipping.", dataset_name)
        return

    output_root = Path(eval_config.output_directory).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, hf_split in dataset_splits.items():
        _process_split(
            dataset_name,
            split_name,
            hf_split,
            clip_encoder=clip_encoder,
            defaults=defaults,
            eval_config=eval_config,
            dataset_spec=dataset_spec,
            output_root=output_root,
        )


def main() -> None:
    ConfigParser.load()
    eval_config = ConfigParser.get(ClipEvaluationConfig)
    defaults = ConfigParser.get(DatasetLoaderDefaultsConfig)
    registry = ConfigParser.get(DatasetRegistryConfig)

    clip_encoder = ClipImageEncoder(
        model_name=eval_config.model_name,
        device=eval_config.device,
        precision=eval_config.precision,  # type: ignore[arg-type]
        normalize_features=eval_config.normalize_features,
    )

    if registry.loader_registry:
        configured_datasets = registry.loader_registry
    else:
        logger.warning("Dataset registry is empty; nothing to process.")
        return

    dataset_selection: Sequence[str]
    if eval_config.dataset_names:
        dataset_selection = list(eval_config.dataset_names)
    else:
        dataset_selection = list(configured_datasets.keys())

    logger.info("Prepping to encode datasets: %s", ", ".join(dataset_selection))

    for dataset_name in dataset_selection:
        try:
            dataset_spec = configured_datasets.get(dataset_name)
            if dataset_spec is None:
                logger.warning("Dataset '%s' not found in registry; skipping.", dataset_name)
                continue
            _process_dataset(
                dataset_name,
                dataset_spec,
                clip_encoder=clip_encoder,
                defaults=defaults,
                eval_config=eval_config,
            )

        except Exception as exc:
            logger.error("Error processing dataset '%s': %s", dataset_name, exc)
            logger.error(traceback.format_exc())
            continue

    logger.info("Completed CLIP embedding extraction for all configured datasets.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
