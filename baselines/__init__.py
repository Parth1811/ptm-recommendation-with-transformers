"""Baseline transferability estimation methods.

Exports all 9 methods used in Model Spider (NeurIPS 2023) for comparing
pre-trained model transferability, plus utilities for running them on
pre-extracted feature files.

Quick start
-----------
>>> from baselines import compute_transferability
>>> score = compute_transferability("LogME", features, labels)

>>> from baselines import run_baseline_from_features
>>> results = run_baseline_from_features("LogME", "artifacts/extracted/datasets")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .transferability_metrics import (
    ALL_METHOD_NAMES,
    METHODS,
    compute_all_transferability,
    compute_transferability,
    gbc,
    h_score,
    lfc,
    leep,
    logme,
    nce,
    nleep,
    otce,
    pactran_dirichlet,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Individual methods
    "h_score", "leep", "logme", "nce", "nleep",
    "otce", "pactran_dirichlet", "gbc", "lfc",
    # Dispatcher + registry
    "compute_transferability", "compute_all_transferability",
    "METHODS", "ALL_METHOD_NAMES",
    # Feature-file utilities
    "run_baseline_from_features", "run_all_baselines_from_features",
]


def run_baseline_from_features(
    method_name: str,
    features_dir: str | Path,
    labels_dir: str | Path | None = None,
    datasets: list[str] | None = None,
) -> dict[str, float]:
    """Load pre-extracted feature files and compute a transferability metric.

    Scans ``features_dir`` for per-dataset subdirectories, loads every ``.npz``
    shard found in ``test/``, ``validation/``, or ``train/`` splits in that
    priority order, then concatenates and runs the chosen method.

    Expected directory layout::

        features_dir/
            {dataset_name}/
                {split}/          # test | validation | train
                    shard_0.npz   # keys: 'features' (N,D), 'labels' (N,)

    Labels may alternatively be stored separately::

        labels_dir/
            {dataset_name}/
                labels.npy        # shape (N,) int

    Args:
        method_name: One of the 9 method names (e.g. ``"LogME"``, ``"H-Score"``)
        features_dir: Root directory containing dataset subdirectories
        labels_dir: Optional separate root for label files
        datasets: Restrict evaluation to these dataset names (default: all found)

    Returns:
        ``{dataset_name: transferability_score}``
    """
    features_dir = Path(features_dir)
    results: dict[str, float] = {}

    if not features_dir.exists():
        logger.warning("Features directory not found: %s", features_dir)
        return results

    for dataset_dir in sorted(d for d in features_dir.iterdir() if d.is_dir()):
        dataset_name = dataset_dir.name
        if datasets is not None and dataset_name not in datasets:
            continue

        # Locate shard files — prefer test, then validation, then train
        npz_files: list[Path] = []
        for split in ("test", "validation", "train", ""):
            split_dir = dataset_dir / split if split else dataset_dir
            if split_dir.exists():
                found = sorted(split_dir.glob("*.npz"))
                if found:
                    npz_files = found
                    break

        if not npz_files:
            continue

        # Load + concatenate all shards
        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as archive:
                if "features" in archive:
                    feats = archive["features"]
                    if feats.ndim == 3:  # (batches, classes, dim) shard format
                        feats = feats.reshape(-1, feats.shape[-1])
                    all_features.append(feats)
                if "labels" in archive:
                    labs = archive["labels"]
                    if labs.ndim == 2:
                        labs = labs.reshape(-1)
                    all_labels.append(labs)

        if not all_features:
            continue

        features = np.concatenate(all_features, axis=0)

        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
        elif labels_dir is not None:
            label_path = Path(labels_dir) / dataset_name / "labels.npy"
            if label_path.exists():
                labels = np.load(label_path)
            else:
                logger.warning("No labels found for %s, skipping", dataset_name)
                continue
        else:
            logger.warning("No labels found for %s, skipping", dataset_name)
            continue

        # Align lengths in case of partial shards
        n = min(len(features), len(labels))
        features = features[:n]
        labels = labels[:n].astype(int)

        try:
            score = compute_transferability(method_name, features, labels)
            results[dataset_name] = score
            logger.info("  %s: %s = %.6f", dataset_name, method_name, score)
        except Exception as exc:
            logger.warning("  %s: %s FAILED (%s)", dataset_name, method_name, exc)

    return results


def run_all_baselines_from_features(
    features_dir: str | Path,
    labels_dir: str | Path | None = None,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Run all (or selected) baseline methods on pre-extracted feature files.

    Args:
        features_dir: Root directory containing dataset subdirectories
        labels_dir: Optional separate root for label files
        datasets: Restrict to these dataset names (default: all found)
        methods: Methods to run (default: all 9 in ``ALL_METHOD_NAMES``)

    Returns:
        ``{method_name: {dataset_name: score}}``
    """
    if methods is None:
        methods = ALL_METHOD_NAMES

    return {
        method: run_baseline_from_features(method, features_dir, labels_dir, datasets)
        for method in methods
    }
