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
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


@lru_cache(maxsize=None)
def _load_dataset_performance_maps() -> Tuple[Dict[str, Dict[str, float]], Dict[str, list[Tuple[str, float]]]]:
    path = BASE_DIR / "constants" / "dataset_model_performance.json"
    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    dataset_map: Dict[str, Dict[str, float]] = {}
    model_map: Dict[str, list[Tuple[str, float]]] = defaultdict(list)

    for dataset_name, entries in raw.items():
        dataset_key = _normalize_name(dataset_name)
        dataset_map.setdefault(dataset_key, {})
        for entry in entries:
            model_key = _normalize_name(entry["model_name"])
            accuracy = float(entry["accuracy"])
            dataset_map[dataset_key][model_key] = accuracy
            model_map[model_key].append((dataset_key, accuracy))

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
        row = matrix.setdefault(dataset_key, {})
        for other_name, value in similarities.items():
            other_key = _normalize_name(other_name)
            row[other_key] = float(value)

    # Ensure symmetry and self-similarity defaults.
    for key_a, row in list(matrix.items()):
        row[key_a] = row.get(key_a, 1.0)
        for key_b, val in row.items():
            matrix.setdefault(key_b, {})
            matrix[key_b][key_a] = matrix[key_b].get(key_a, val)
            matrix[key_b][key_b] = matrix[key_b].get(key_b, 1.0)

    return matrix


def _resolve_key(name: str, candidates: Iterable[str]) -> str:
    name = _normalize_name(name)
    candidate_list = list(candidates)
    if name in candidate_list:
        return name
    for candidate in candidate_list:
        if name in candidate or candidate in name:
            return candidate
    return name


def _get_similarity(source: str, target: str, matrix: Mapping[str, Mapping[str, float]]) -> float:
    if not matrix:
        return 1.0 if _normalize_name(source) == _normalize_name(target) else 0.0

    keys = matrix.keys()
    key_a = _resolve_key(source, keys)
    key_b = _resolve_key(target, keys)

    score = matrix.get(key_a, {}).get(key_b)
    if score is None:
        score = matrix.get(key_b, {}).get(key_a)
    if score is None:
        return 1.0 if key_a == key_b else 0.0
    return float(score)


def _compute_weighted_score(
    target_dataset: str,
    model_key: str,
    model_performance: Mapping[str, Sequence[Tuple[str, float]]],
    similarity_matrix: Mapping[str, Mapping[str, float]],
) -> float:
    entries = model_performance.get(model_key)
    if not entries:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for other_dataset, accuracy in entries:
        weight = _get_similarity(target_dataset, other_dataset, similarity_matrix)
        if weight > 0.0:
            weighted_sum += weight * accuracy
            total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def compute_true_ranks(dataset_name: str, model_names: Sequence[str]) -> torch.Tensor:
    """Return 1-based ranking tensor for the provided models on the given dataset."""

    dataset_map, model_map = _load_dataset_performance_maps()
    similarity_matrix = _load_similarity_matrix()

    dataset_key = _resolve_key(dataset_name, dataset_map.keys())

    scores = []
    for model_name in model_names:
        model_key = _normalize_name(model_name)
        direct_score = dataset_map.get(dataset_key, {}).get(model_key)
        if direct_score is not None:
            score = float(direct_score)
        else:
            score = _compute_weighted_score(dataset_key, model_key, model_map, similarity_matrix)
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
