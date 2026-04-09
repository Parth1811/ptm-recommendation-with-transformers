"""Benchmark evaluation against Model Spider baselines.

Parses pre-computed baseline metrics from Model Spider (NeurIPS 2023) and
computes ranking correlation metrics (τ_w, Precision@K, NDCG, RelTop1) for
each method on each dataset. This enables direct comparison of Cross-Select
against 9 established transferability estimation methods.

Reference:
    Zhang et al., "Model Spider: Learning to Rank Pre-Trained Models
    Efficiently", NeurIPS 2023.

Data source:
    https://github.com/zhangyikaii/Model-Spider/blob/main/assests/baseline_results.csv

For evaluation of trained Cross-Select / SelfAttn models, see:
    evaluate_models.py

For running transferability baselines on pre-extracted features, see:
    baselines.run_baseline_from_features()
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ZOO_ORDER = [
    "googlenet",
    "inception_v3",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet169",
    "densenet201",
    "mobilenet_v2",
    "mnasnet1_0",
]

# Test datasets used by Model Spider (single-source zoo)
TEST_DATASETS = [
    "Aircraft",
    "Cars",
    "Caltech101",
    "CIFAR10",
    "CIFAR100",
    "DTD",
    "Pet",
    "SUN397",
]

# Training datasets for Cross-Select (ground truth must be computed here)
TRAIN_DATASETS = [
    "EuroSAT",
    "OfficeHome",
    "PACS",
    "SmallNORB",
    "STL10",
    "VLCS",
]

# Ground truth fine-tuning accuracy from Model Spider's learnware_info.py
# Order matches MODEL_ZOO_ORDER:
# [googlenet, inception_v3, resnet50, resnet101, resnet152,
#  densenet121, densenet169, densenet201, mobilenet_v2, mnasnet1_0]
GROUND_TRUTH_FINETUNING: dict[str, list[float]] = {
    "Aircraft":  [82.7, 88.8, 86.6, 85.6, 85.3, 85.4, 84.5, 84.6, 82.8, 72.8],
    "Caltech101":[91.7, 94.3, 91.8, 93.1, 93.2, 91.9, 92.5, 93.4, 89.1, 91.5],
    "CIFAR10":   [96.2, 97.5, 96.8, 97.7, 97.9, 97.2, 97.4, 97.4, 95.7, 96.8],
    "CIFAR100":  [83.2, 86.6, 84.5, 87.0, 87.6, 84.8, 85.0, 86.0, 80.8, 83.9],
    "Cars":      [91.0, 92.3, 91.7, 91.7, 92.0, 91.5, 91.5, 91.0, 91.0, 88.5],
    "DTD":       [73.6, 77.2, 75.2, 76.2, 75.4, 74.9, 74.8, 74.5, 72.9, 72.8],
    "Pet":       [91.9, 93.5, 92.5, 94.0, 94.5, 92.9, 93.1, 92.8, 90.5, 89.4],
    "SUN397":    [62.0, 65.7, 64.7, 64.8, 66.0, 62.3, 63.0, 64.7, 60.5, 60.7],
}

# Multi-source experiment constants
MULTI_SOURCE_ARCHITECTURES = ["resnet50", "inception_v3", "densenet201"]
MULTI_SOURCE_PRETRAIN_DATASETS = [
    "AID", "CIFAR10", "CIFAR100", "Caltech101", "Cars", "Dogs",
    "EuroSAT", "Flowers", "Food", "NABirds", "PACS", "Resisc45",
    "SmallNORB", "SUN397",
]
MULTI_SOURCE_TEST_DATASETS = [
    "Aircraft", "CUB2011", "DTD", "Pet", "STL10", "VLCS", "AID",
]

# Canonical ordered list of the 9 baseline methods (matches CSV column order)
BASELINE_METHODS: list[str] = [
    "H-Score",
    "LEEP",
    "LogME",
    "NCE",
    "NLEEP",
    "OTCE",
    "PACTranDirichlet",
    "GBC",
    "LFC",
]

# All methods have higher = better (scores are already negated where needed)
HIGHER_IS_BETTER: dict[str, bool] = {m: True for m in BASELINE_METHODS}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetResults:
    """Fine-tuned accuracy and baseline metric scores for one dataset."""

    dataset_name: str
    # model_name -> fine-tuned accuracy (ground truth)
    ground_truth_accuracy: dict[str, float] = field(default_factory=dict)
    # method_name -> {model_name -> score}
    method_scores: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Ranking correlation result for one method on one dataset."""

    dataset_name: str
    method_name: str
    tau_w: float
    tau_b: float
    spearman_rho: float
    ndcg: float
    precision_at_1: float
    precision_at_3: float
    relative_top1: float


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_model_spider_csv(csv_path: str | Path) -> dict[str, DatasetResults]:
    """Parse Model Spider's baseline_results.csv into structured data.

    The CSV has columns: Dataset, Model, H-Score, LEEP, LogME, NCE, NLEEP,
    OTCE, PACTranDirichlet, GBC, LFC.

    The H-Score column contains the actual fine-tuned accuracy (ground truth).

    Args:
        csv_path: Path to baseline_results.csv

    Returns:
        Dictionary mapping dataset_name -> DatasetResults
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline results file not found: {csv_path}")

    results: dict[str, DatasetResults] = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["Dataset"].strip()
            model = row["Model"].strip()

            if dataset not in results:
                results[dataset] = DatasetResults(dataset_name=dataset)

            # H-Score column == ground-truth fine-tuned accuracy in Model Spider
            results[dataset].ground_truth_accuracy[model] = float(row["H-Score"])

            for method in BASELINE_METHODS:
                if method not in results[dataset].method_scores:
                    results[dataset].method_scores[method] = {}
                results[dataset].method_scores[method][model] = float(row[method])

    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _calculate_ndcg(true_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """Inline NDCG@full-list (avoids importing evaluate_ranking + beautilog)."""
    n = len(true_ranks)
    true_relevance = n - true_ranks.astype(float)  # higher original rank → higher relevance
    pred_order = np.argsort(pred_ranks)             # sorted by predicted rank (ascending)
    dcg = sum(true_relevance[idx] / np.log2(i + 2) for i, idx in enumerate(pred_order))
    ideal_order = np.argsort(true_ranks)
    idcg = sum(true_relevance[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0.0


def compute_ranking_from_scores(
    ground_truth: dict[str, float],
    predicted_scores: dict[str, float],
    higher_is_better: bool = True,
) -> tuple[list[int], list[int]]:
    """Convert raw scores to 0-indexed rank orderings (0 = best model).

    Args:
        ground_truth: model_name -> ground truth accuracy
        predicted_scores: model_name -> predicted transferability score
        higher_is_better: If True, higher predicted scores rank first

    Returns:
        (true_ranks, predicted_ranks) — parallel lists of integers
    """
    models = sorted(ground_truth.keys())
    gt_values = [ground_truth[m] for m in models]
    pred_values = [predicted_scores[m] for m in models]

    # Ground truth: accuracy is always higher-is-better
    gt_order = np.argsort(gt_values)[::-1]
    pred_order = np.argsort(pred_values)[::-1] if higher_is_better else np.argsort(pred_values)

    gt_ranks = np.zeros(len(models), dtype=int)
    pred_ranks = np.zeros(len(models), dtype=int)
    for rank, idx in enumerate(gt_order):
        gt_ranks[idx] = rank
    for rank, idx in enumerate(pred_order):
        pred_ranks[idx] = rank

    return gt_ranks.tolist(), pred_ranks.tolist()


def compute_precision_at_k(
    true_ranks: list[int], pred_ranks: list[int], k: int
) -> float:
    """Precision@K: is the true best model within the top-K predicted?

    Args:
        true_ranks: Ground truth rank per model (0 = best)
        pred_ranks: Predicted rank per model (0 = best)
        k: Top-K window

    Returns:
        1.0 if the true best is in the top-K predictions, else 0.0
    """
    true_best_idx = int(np.argmin(true_ranks))
    return 1.0 if pred_ranks[true_best_idx] < k else 0.0


def compute_relative_top1(
    ground_truth: dict[str, float],
    pred_ranks: list[int],
    models: list[str],
) -> float:
    """Relative Top-1: accuracy(recommended) / accuracy(best).

    Args:
        ground_truth: model_name -> ground truth accuracy
        pred_ranks: Predicted ranks for each model (parallel to models)
        models: Ordered list of model names

    Returns:
        Ratio of the recommended model's accuracy to the best model's accuracy
    """
    recommended_model = models[int(np.argmin(pred_ranks))]
    best_accuracy = max(ground_truth.values())
    recommended_accuracy = ground_truth[recommended_model]
    return recommended_accuracy / best_accuracy if best_accuracy > 0 else 0.0


def _compute_benchmark_result(
    dataset_name: str,
    method_name: str,
    gt: dict[str, float],
    scores: dict[str, float],
    higher_is_better: bool = True,
) -> BenchmarkResult:
    """Compute all ranking metrics for a single (dataset, method) pair."""
    models = sorted(gt.keys())
    gt_values = np.array([gt[m] for m in models])
    pred_values = np.array([scores[m] for m in models])

    true_ranks, pred_ranks = compute_ranking_from_scores(
        gt, scores, higher_is_better=higher_is_better
    )

    try:
        tau_w, _ = stats.weightedtau(gt_values, pred_values)
    except Exception:
        tau_w = 0.0

    try:
        tau_b, _ = stats.kendalltau(gt_values, pred_values)
    except Exception:
        tau_b = 0.0

    try:
        rho, _ = stats.spearmanr(gt_values, pred_values)
    except Exception:
        rho = 0.0

    return BenchmarkResult(
        dataset_name=dataset_name,
        method_name=method_name,
        tau_w=float(tau_w),
        tau_b=float(tau_b),
        spearman_rho=float(rho),
        ndcg=_calculate_ndcg(np.array(true_ranks), np.array(pred_ranks)),
        precision_at_1=compute_precision_at_k(true_ranks, pred_ranks, k=1),
        precision_at_3=compute_precision_at_k(true_ranks, pred_ranks, k=3),
        relative_top1=compute_relative_top1(gt, pred_ranks, models),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all_baselines(csv_path: str | Path) -> list[BenchmarkResult]:
    """Evaluate all 9 baseline methods from a Model Spider CSV.

    Args:
        csv_path: Path to Model Spider's baseline_results.csv

    Returns:
        List of BenchmarkResult for each (dataset, method) pair
    """
    dataset_results = parse_model_spider_csv(csv_path)
    all_results: list[BenchmarkResult] = []

    for dataset_name, dr in dataset_results.items():
        gt = dr.ground_truth_accuracy
        for method_name, scores in dr.method_scores.items():
            higher = HIGHER_IS_BETTER.get(method_name, True)
            all_results.append(
                _compute_benchmark_result(dataset_name, method_name, gt, scores, higher)
            )

    return all_results


# ---------------------------------------------------------------------------
# Display / persistence
# ---------------------------------------------------------------------------

def print_comparison_table(
    results: list[BenchmarkResult],
    extra_results: list[BenchmarkResult] | None = None,
) -> None:
    """Print τ_w and Precision@1 comparison tables.

    Args:
        results: Baseline BenchmarkResults
        extra_results: Optional additional results (Cross-Select, SelfAttn, etc.)
    """
    all_results = results + (extra_results or [])

    method_dataset_tau: dict[str, dict[str, float]] = defaultdict(dict)
    method_dataset_p1: dict[str, dict[str, float]] = defaultdict(dict)
    datasets_seen: set[str] = set()
    methods_seen: list[str] = list(BASELINE_METHODS)

    for r in all_results:
        method_dataset_tau[r.method_name][r.dataset_name] = r.tau_w
        method_dataset_p1[r.method_name][r.dataset_name] = r.precision_at_1
        datasets_seen.add(r.dataset_name)
        if r.method_name not in methods_seen:
            methods_seen.append(r.method_name)

    datasets = sorted(datasets_seen)
    header = f"{'Method':<20}" + "".join(f"{d:>12}" for d in datasets) + f"{'Mean':>12}"
    sep = "=" * len(header)

    for title, data in [("Weighted Kendall τ_w", method_dataset_tau),
                        ("Precision@1", method_dataset_p1)]:
        print(f"\n{sep}\nBENCHMARK COMPARISON — {title}\n{sep}")
        print(header)
        print("-" * len(header))

        best_mean, best_method = -float("inf"), ""
        for method in methods_seen:
            values = [data[method].get(d, float("nan")) for d in datasets]
            valid = [v for v in values if not np.isnan(v)]
            mean_val = float(np.mean(valid)) if valid else float("nan")
            fmt = ".1%" if "Precision" in title else ".4f"
            row = f"{method:<20}"
            for v in values:
                row += f"{'—':>12}" if np.isnan(v) else f"{v:>12{fmt}}"
            row += f"{mean_val:>12{fmt}}" if not np.isnan(mean_val) else f"{'—':>12}"
            print(row)
            if mean_val > best_mean:
                best_mean, best_method = mean_val, method

        print("-" * len(header))
        fmt = ".1%" if "Precision" in title else ".4f"
        print(f"Best: {best_method} ({best_mean:{fmt}})")


def save_results_csv(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """Save benchmark results to CSV.

    Args:
        results: List of BenchmarkResult
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset_name", "method_name", "tau_w", "tau_b", "spearman_rho",
        "ndcg", "precision_at_1", "precision_at_3", "relative_top1",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset_name": r.dataset_name,
                "method_name": r.method_name,
                "tau_w": f"{r.tau_w:.6f}",
                "tau_b": f"{r.tau_b:.6f}",
                "spearman_rho": f"{r.spearman_rho:.6f}",
                "ndcg": f"{r.ndcg:.6f}",
                "precision_at_1": f"{r.precision_at_1:.1f}",
                "precision_at_3": f"{r.precision_at_3:.1f}",
                "relative_top1": f"{r.relative_top1:.6f}",
            })

    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    csv_path: str | Path = "constants/model_spider_baseline_results.csv",
    output_path: str | Path = "artifacts/benchmark_comparison.csv",
) -> None:
    """Run benchmark evaluation and print comparison table.

    Args:
        csv_path: Path to Model Spider's baseline_results.csv
        output_path: Path to save detailed results CSV
    """
    print("Loading Model Spider baseline results...")
    results = evaluate_all_baselines(csv_path)
    print(f"Evaluated {len(results)} (dataset, method) combinations")
    print_comparison_table(results)
    save_results_csv(results, output_path)


if __name__ == "__main__":
    import sys
    _csv = sys.argv[1] if len(sys.argv) > 1 else "constants/model_spider_baseline_results.csv"
    _out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/benchmark_comparison.csv"
    main(_csv, _out)
