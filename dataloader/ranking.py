"""Utilities for deriving dataset/model rankings using stored metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


BASE_DIR = Path(__file__).resolve().parents[1]


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_").replace("_embedding", "").strip()


@lru_cache(maxsize=None)
def _load_dataset_performance_maps() -> Tuple[Dict[str, list[Tuple[str, float]]], Dict[str, str]]:
    path = BASE_DIR / "constants" / "dataset_model_performance.json"
    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    dataset_map: Dict[str, Dict[str, float]] = {}
    model_map: Dict[str, str] = {}

    for dataset_name, entries in raw.items():
        dataset_key = _normalize_name(dataset_name)
        dataset_map[dataset_key] = {
            _normalize_name(entry["model_name"]): float(entry["accuracy"])
            for entry in entries
        }

        for entry in entries:
            model_name = _normalize_name(entry["model_name"])
            model_map[model_name] = dataset_key

    return dataset_map, model_map


@lru_cache(maxsize=None)
def _load_similarity_matrix() -> Dict[str, Dict[str, float]]:
    path = BASE_DIR / "constants" / "dataset_similarity.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    matrix: Dict[str, Dict[str, float]] = {}

    for dataset_name, similarities in raw.items():
        dataset_key = _normalize_name(dataset_name)
        matrix[dataset_key] = { _normalize_name(other_dataset): float(score) for other_dataset, score in similarities.items() }
        # matrix[dataset_key] = dict(sorted(matrix[dataset_key].items(), key=lambda x: x[1], reverse=True))

    return matrix



def _compute_weighted_score(
    target_dataset: str,
    model_key: str,
    dataset_map: Mapping[str, Sequence[Tuple[str, float]]],
    model_map: Mapping[str, str],
    similarity_matrix: Mapping[str, Sequence[Tuple[str, float]]],
) -> float:

    model_base_dataset = model_map.get(model_key)
    if model_base_dataset is None:
        return 0.0

    similarity_scores = similarity_matrix.get(target_dataset, {}).get(model_base_dataset, 0.0)

    return dataset_map.get(model_base_dataset, {}).get(model_key, 0.0) * similarity_scores


def compute_true_ranks(dataset_name: str, model_names: Sequence[str]) -> torch.Tensor:
    """Return 1-based ranking tensor for the provided models on the given dataset."""

    dataset_map, model_map = _load_dataset_performance_maps()
    similarity_matrix = _load_similarity_matrix()

    dataset_key = _normalize_name(dataset_name)

    scores = []
    for model_name in model_names:
        model_key = _normalize_name(model_name)
        direct_score = dataset_map.get(dataset_key, {}).get(model_key)
        if direct_score is not None:
            score = float(direct_score) + 1
        else:
            score = _compute_weighted_score(dataset_key, model_key, dataset_map, model_map, similarity_matrix)
        scores.append(score)

    if not scores:
        return torch.tensor([], dtype=torch.long)

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    tie_breaker = torch.arange(len(scores), dtype=torch.float32) * 1e-6
    augmented = score_tensor + tie_breaker

    sorted_indices = torch.argsort(augmented, descending=True)
    ranks = torch.empty_like(sorted_indices, dtype=torch.long)
    ranks[sorted_indices] = torch.arange(1, len(scores) + 1, dtype=torch.long)
    return ranks


__all__ = ["compute_true_ranks"]
