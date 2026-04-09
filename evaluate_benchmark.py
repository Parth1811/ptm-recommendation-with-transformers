"""Benchmark evaluation against Model Spider baselines.

Parses pre-computed baseline metrics from Model Spider (NeurIPS 2023) and
computes ranking correlation metrics (τ_w, Precision@K) for each method
on each dataset. This enables direct comparison of Cross-Select against
9 established transferability estimation methods.

Reference:
    Zhang et al., "Model Spider: Learning to Rank Pre-Trained Models
    Efficiently", NeurIPS 2023.

Data source:
    https://github.com/zhangyikaii/Model-Spider/blob/main/assests/baseline_results.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _calculate_ndcg(true_ranks: np.ndarray, pred_ranks: np.ndarray) -> float:
    """Inline NDCG calculation (avoids importing evaluate_ranking + beautilog)."""
    n = len(true_ranks)
    k = n
    true_relevance = n - true_ranks.astype(float)
    pred_order = np.argsort(pred_ranks)[:k]
    dcg = sum(
        true_relevance[idx] / np.log2(i + 2) for i, idx in enumerate(pred_order)
    )
    ideal_order = np.argsort(true_ranks)[:k]
    idcg = sum(
        true_relevance[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_order)
    )
    return dcg / idcg if idcg > 0 else 0.0


# Ground truth fine-tuning accuracy for Model Spider's 10 single-source models.
# Columns from baseline_results.csv: first numeric column is H-Score (= accuracy).
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

# Training datasets for Model Spider (ground truth must be computed here)
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
GROUND_TRUTH_FINETUNING = {
    "Aircraft": [82.7, 88.8, 86.6, 85.6, 85.3, 85.4, 84.5, 84.6, 82.8, 72.8],
    "Caltech101": [91.7, 94.3, 91.8, 93.1, 93.2, 91.9, 92.5, 93.4, 89.1, 91.5],
    "CIFAR10": [96.2, 97.5, 96.8, 97.7, 97.9, 97.2, 97.4, 97.4, 95.7, 96.8],
    "CIFAR100": [83.2, 86.6, 84.5, 87.0, 87.6, 84.8, 85.0, 86.0, 80.8, 83.9],
    "Cars": [91.0, 92.3, 91.7, 91.7, 92.0, 91.5, 91.5, 91.0, 91.0, 88.5],
    "DTD": [73.6, 77.2, 75.2, 76.2, 75.4, 74.9, 74.8, 74.5, 72.9, 72.8],
    "Pet": [91.9, 93.5, 92.5, 94.0, 94.5, 92.9, 93.1, 92.8, 90.5, 89.4],
    "SUN397": [62.0, 65.7, 64.7, 64.8, 66.0, 62.3, 63.0, 64.7, 60.5, 60.7],
}

# Multi-source experiment: 42 models = 3 archs × 14 source datasets
# 3 architectures: ResNet-50, Inception-V3, DenseNet-201
# 14 source pre-training datasets (after SELECTED_DATASETS filtering)
MULTI_SOURCE_ARCHITECTURES = ["resnet50", "inception_v3", "densenet201"]
MULTI_SOURCE_PRETRAIN_DATASETS = [
    "AID", "CIFAR10", "CIFAR100", "Caltech101", "Cars", "Dogs",
    "EuroSAT", "Flowers", "Food", "NABirds", "PACS", "Resisc45",
    "SmallNORB", "SUN397",
]
MULTI_SOURCE_TRAIN_TASK_DATASETS = MULTI_SOURCE_PRETRAIN_DATASETS  # 14 base datasets used in t127-t222
MULTI_SOURCE_TEST_DATASETS = [
    "Aircraft", "CUB2011", "DTD", "Pet", "STL10", "VLCS", "AID",
]

# Baseline method names in order of columns in the CSV
BASELINE_METHODS = [
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


def parse_model_spider_csv(csv_path: str | Path) -> dict[str, DatasetResults]:
    """Parse Model Spider's baseline_results.csv into structured data.

    The CSV has columns: Dataset, Model, H-Score, LEEP, LogME, NCE, NLEEP, OTCE,
    PACTranDirichlet, GBC, LFC.

    H-Score column contains the actual fine-tuned accuracy (used as ground truth).

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

            # H-Score is the fine-tuned accuracy (ground truth)
            gt_accuracy = float(row["H-Score"])
            results[dataset].ground_truth_accuracy[model] = gt_accuracy

            # Store each baseline method's score
            for method in BASELINE_METHODS:
                if method not in results[dataset].method_scores:
                    results[dataset].method_scores[method] = {}
                results[dataset].method_scores[method][model] = float(row[method])

    return results


def compute_ranking_from_scores(
    ground_truth: dict[str, float],
    predicted_scores: dict[str, float],
    higher_is_better: bool = True,
) -> tuple[list[int], list[int]]:
    """Convert raw scores to rank orderings for comparison.

    Args:
        ground_truth: model_name -> ground truth accuracy
        predicted_scores: model_name -> predicted transferability score
        higher_is_better: If True, higher scores predict better performance

    Returns:
        Tuple of (true_ranks, predicted_ranks) as lists of integers
    """
    # Ensure consistent model ordering
    models = sorted(ground_truth.keys())

    # Get values in consistent order
    gt_values = [ground_truth[m] for m in models]
    pred_values = [predicted_scores[m] for m in models]

    # Convert to ranks (0-indexed, lower is better = higher original value)
    gt_order = np.argsort(gt_values)[::-1] if True else np.argsort(gt_values)
    pred_order = (
        np.argsort(pred_values)[::-1]
        if higher_is_better
        else np.argsort(pred_values)
    )

    # Create rank arrays
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
    """Compute Precision@K: is the actual best model in the top-K predictions?

    Args:
        true_ranks: Ground truth rank for each model (0 = best)
        pred_ranks: Predicted rank for each model (0 = best)
        k: Number of top positions to consider

    Returns:
        1.0 if the true best model is in the top-K predicted, 0.0 otherwise
    """
    true_array = np.array(true_ranks)
    pred_array = np.array(pred_ranks)

    # Find the index of the true best model (rank 0)
    true_best_idx = np.argmin(true_array)

    # Check if this model is in the top-K predictions
    return 1.0 if pred_array[true_best_idx] < k else 0.0


def compute_relative_top1(
    ground_truth: dict[str, float], pred_ranks: list[int], models: list[str]
) -> float:
    """Compute Relative Top-1: accuracy(recommended) / accuracy(best).

    Args:
        ground_truth: model_name -> ground truth accuracy
        pred_ranks: Predicted ranks for each model
        models: Ordered list of model names

    Returns:
        Ratio of recommended model's accuracy to best model's accuracy
    """
    pred_array = np.array(pred_ranks)
    recommended_idx = np.argmin(pred_array)
    recommended_model = models[recommended_idx]

    best_accuracy = max(ground_truth.values())
    recommended_accuracy = ground_truth[recommended_model]

    return recommended_accuracy / best_accuracy if best_accuracy > 0 else 0.0


# Which baseline methods have higher=better vs lower=better for scores
# Determined from the metric definitions:
HIGHER_IS_BETTER = {
    "H-Score": True,  # Actual accuracy - always higher is better
    "LEEP": True,     # Log Expected Empirical Prediction - higher is better
    "LogME": True,    # Log Maximum Evidence - higher is better
    "NCE": True,      # Negative Conditional Entropy - higher (less negative) is better
    "NLEEP": True,    # Normalized LEEP - higher is better
    "OTCE": True,     # OT-based Conditional Entropy - higher is better
    "PACTranDirichlet": True,  # PAC-Bayesian bound - higher (less negative) is better
    "GBC": True,      # Gaussian Bhattacharyya - higher is better
    "LFC": True,      # Linear Feature Correlation - higher is better
}


def evaluate_all_baselines(
    csv_path: str | Path,
) -> list[BenchmarkResult]:
    """Evaluate all baseline methods and compute ranking metrics.

    Args:
        csv_path: Path to Model Spider's baseline_results.csv

    Returns:
        List of BenchmarkResult for each (dataset, method) pair
    """
    dataset_results = parse_model_spider_csv(csv_path)
    all_results: list[BenchmarkResult] = []

    for dataset_name, dr in dataset_results.items():
        gt = dr.ground_truth_accuracy
        models = sorted(gt.keys())

        for method_name, scores in dr.method_scores.items():
            higher = HIGHER_IS_BETTER.get(method_name, True)
            true_ranks, pred_ranks = compute_ranking_from_scores(
                gt, scores, higher_is_better=higher
            )

            # Compute τ_w using scipy
            gt_values = [gt[m] for m in models]
            pred_values = [scores[m] for m in models]

            try:
                tau_w, _ = stats.weightedtau(
                    np.array(gt_values), np.array(pred_values)
                )
            except Exception:
                tau_w = 0.0

            try:
                tau_b, _ = stats.kendalltau(
                    np.array(gt_values), np.array(pred_values)
                )
            except Exception:
                tau_b = 0.0

            try:
                rho, _ = stats.spearmanr(
                    np.array(gt_values), np.array(pred_values)
                )
            except Exception:
                rho = 0.0

            # NDCG
            ndcg = _calculate_ndcg(
                np.array(true_ranks), np.array(pred_ranks)
            )

            # Precision@K
            p_at_1 = compute_precision_at_k(true_ranks, pred_ranks, k=1)
            p_at_3 = compute_precision_at_k(true_ranks, pred_ranks, k=3)

            # Relative Top-1
            rel_top1 = compute_relative_top1(gt, pred_ranks, models)

            all_results.append(
                BenchmarkResult(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    tau_w=tau_w,
                    tau_b=tau_b,
                    spearman_rho=rho,
                    ndcg=ndcg,
                    precision_at_1=p_at_1,
                    precision_at_3=p_at_3,
                    relative_top1=rel_top1,
                )
            )

    return all_results


def print_comparison_table(
    results: list[BenchmarkResult],
    cross_select_results: dict[str, BenchmarkResult] | None = None,
) -> None:
    """Print the primary comparison table (τ_w per dataset per method).

    Args:
        results: List of baseline BenchmarkResults
        cross_select_results: Optional dict of dataset_name -> CrossSelect result
    """
    # Group by method
    method_dataset_tau: dict[str, dict[str, float]] = defaultdict(dict)
    datasets_seen = set()

    for r in results:
        method_dataset_tau[r.method_name][r.dataset_name] = r.tau_w
        datasets_seen.add(r.dataset_name)

    if cross_select_results:
        for ds, r in cross_select_results.items():
            method_dataset_tau["Cross-Select"][ds] = r.tau_w
            datasets_seen.add(ds)

    datasets = sorted(datasets_seen)
    methods = BASELINE_METHODS.copy()
    if cross_select_results:
        methods.append("Cross-Select")

    # Print header
    header = f"{'Method':<20}" + "".join(f"{d:>12}" for d in datasets) + f"{'Mean':>12}"
    print("\n" + "=" * len(header))
    print("BENCHMARK COMPARISON — Weighted Kendall τ_w")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Print each method row
    best_mean = -float("inf")
    best_method = ""
    for method in methods:
        values = [method_dataset_tau[method].get(d, float("nan")) for d in datasets]
        valid_values = [v for v in values if not np.isnan(v)]
        mean_val = np.mean(valid_values) if valid_values else float("nan")
        row = f"{method:<20}"
        for v in values:
            if np.isnan(v):
                row += f"{'—':>12}"
            else:
                row += f"{v:>12.4f}"
        row += f"{mean_val:>12.4f}"
        print(row)
        if mean_val > best_mean:
            best_mean = mean_val
            best_method = method

    print("-" * len(header))
    print(f"\nBest method by mean τ_w: {best_method} ({best_mean:.4f})")

    # Precision@1 table
    print("\n" + "=" * len(header))
    print("BENCHMARK COMPARISON — Precision@1")
    print("=" * len(header))

    method_dataset_p1: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        method_dataset_p1[r.method_name][r.dataset_name] = r.precision_at_1
    if cross_select_results:
        for ds, r in cross_select_results.items():
            method_dataset_p1["Cross-Select"][ds] = r.precision_at_1

    print(header.replace("τ_w", "P@1"))
    print("-" * len(header))
    for method in methods:
        values = [method_dataset_p1[method].get(d, float("nan")) for d in datasets]
        valid_values = [v for v in values if not np.isnan(v)]
        mean_val = np.mean(valid_values) if valid_values else float("nan")
        row = f"{method:<20}"
        for v in values:
            if np.isnan(v):
                row += f"{'—':>12}"
            else:
                row += f"{v:>12.1%}"
        row += f"{mean_val:>12.1%}"
        print(row)
    print("-" * len(header))


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
        "dataset_name",
        "method_name",
        "tau_w",
        "tau_b",
        "spearman_rho",
        "ndcg",
        "precision_at_1",
        "precision_at_3",
        "relative_top1",
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

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "constants/model_spider_baseline_results.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/benchmark_comparison.csv"
    main(csv_path, output_path)


# ---------------------------------------------------------------------------
# Cross-Select / Self-Attention Baseline evaluation on Model Spider benchmark
# ---------------------------------------------------------------------------

def _load_model_and_predict(
    model_class: str,
    checkpoint_path: str | Path,
    dataset_tokens_dir: str | Path,
    model_embeddings_dir: str | Path,
    test_datasets: list[str] | None = None,
    device: str = "cuda",
) -> dict[str, dict[str, float]]:
    """Load a trained model and compute predicted ranking scores for each test dataset.

    Args:
        model_class: "cross_select" or "self_attention"
        checkpoint_path: Path to .pt checkpoint
        dataset_tokens_dir: Root dir containing dataset token shards
        model_embeddings_dir: Root dir containing model embedding .npz files
        test_datasets: List of test dataset names (default: TEST_DATASETS)
        device: torch device

    Returns:
        {dataset_name: {model_name: predicted_score}} for each test dataset
    """
    import torch
    from pathlib import Path as _Path

    checkpoint_path = _Path(checkpoint_path)
    dataset_tokens_dir = _Path(dataset_tokens_dir)
    model_embeddings_dir = _Path(model_embeddings_dir)
    test_datasets = test_datasets or TEST_DATASETS

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model
    if model_class == "cross_select":
        from model import CustomSimilarityTransformer
        model = CustomSimilarityTransformer()
    elif model_class == "self_attention":
        from model import SelfAttentionBaseline
        model = SelfAttentionBaseline()
    else:
        raise ValueError(f"Unknown model_class: {model_class}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load model embeddings
    model_files = sorted(f for f in model_embeddings_dir.rglob("*.npz") if f.is_file())
    model_embeddings = []
    model_names = []
    for fpath in model_files:
        with np.load(fpath) as archive:
            emb = archive["embedding"]
        t = torch.tensor(emb, dtype=torch.float32)
        if t.ndim > 1:
            t = t.reshape(-1)
        model_embeddings.append(t)
        model_names.append(fpath.stem)

    model_tokens = torch.stack(model_embeddings, dim=0).unsqueeze(0).to(device)  # (1, N, D)

    # Run inference per test dataset
    all_predictions = {}

    for dataset_name in test_datasets:
        dataset_dir = dataset_tokens_dir / dataset_name
        if not dataset_dir.exists():
            print(f"  Warning: dataset tokens not found for {dataset_name}, skipping")
            continue

        # Find shard files (try test split first, then validation, then train)
        shard_files = []
        for split in ["test", "validation", "train"]:
            split_dir = dataset_dir / split
            if split_dir.exists():
                shard_files = sorted(split_dir.glob("*.npz"))
                if shard_files:
                    break

        if not shard_files:
            print(f"  Warning: no shard files found for {dataset_name}, skipping")
            continue

        # Aggregate scores across shards
        all_scores = []
        with torch.no_grad():
            for shard_path in shard_files:
                with np.load(shard_path, allow_pickle=True) as archive:
                    features = archive["features"]  # (batches, num_classes, dim)

                dataset_tokens_tensor = torch.tensor(features, dtype=torch.float32).to(device)

                # Expand model tokens to match batch size
                batch_size = dataset_tokens_tensor.shape[0]
                m_tokens = model_tokens.expand(batch_size, -1, -1)

                # Forward pass
                if model_class == "cross_select":
                    scores = model(m_tokens, dataset_tokens_tensor)  # (B, N)
                else:
                    probs, scores = model(m_tokens, dataset_tokens_tensor, return_attention_weights=True)

                all_scores.append(scores.cpu())

        # Average scores across all batches/shards
        avg_scores = torch.cat(all_scores, dim=0).mean(dim=0)  # (N,)

        # Map to model names
        all_predictions[dataset_name] = {
            model_names[i]: avg_scores[i].item()
            for i in range(len(model_names))
        }

    return all_predictions


def run_cross_select_evaluation(
    checkpoint_path: str | Path,
    dataset_tokens_dir: str | Path = "artifacts/extracted/datasets",
    model_embeddings_dir: str | Path = "artifacts/extracted/model_embeddings",
    csv_path: str | Path = "constants/model_spider_baseline_results.csv",
    test_datasets: list[str] | None = None,
    device: str = "cuda",
) -> list[BenchmarkResult]:
    """Evaluate a trained Cross-Select model on the Model Spider benchmark.

    Loads the checkpoint, runs inference on each test dataset, and computes
    the same ranking metrics as the baselines for direct comparison.

    Args:
        checkpoint_path: Path to trained CustomSimilarityTransformer .pt file
        dataset_tokens_dir: Root dir for dataset token shards
        model_embeddings_dir: Root dir for model embedding .npz files
        csv_path: Path to baseline_results.csv (for ground truth)
        test_datasets: Override test datasets (default: TEST_DATASETS)
        device: torch device

    Returns:
        List of BenchmarkResult for Cross-Select on each test dataset
    """
    print("Running Cross-Select evaluation...")
    return _evaluate_trained_model(
        model_class="cross_select",
        method_name="Cross-Select",
        checkpoint_path=checkpoint_path,
        dataset_tokens_dir=dataset_tokens_dir,
        model_embeddings_dir=model_embeddings_dir,
        csv_path=csv_path,
        test_datasets=test_datasets,
        device=device,
    )


def run_self_attention_evaluation(
    checkpoint_path: str | Path,
    dataset_tokens_dir: str | Path = "artifacts/extracted/datasets",
    model_embeddings_dir: str | Path = "artifacts/extracted/model_embeddings",
    csv_path: str | Path = "constants/model_spider_baseline_results.csv",
    test_datasets: list[str] | None = None,
    device: str = "cuda",
) -> list[BenchmarkResult]:
    """Evaluate a trained SelfAttentionBaseline on the Model Spider benchmark.

    Args:
        checkpoint_path: Path to trained SelfAttentionBaseline .pt file
        dataset_tokens_dir: Root dir for dataset token shards
        model_embeddings_dir: Root dir for model embedding .npz files
        csv_path: Path to baseline_results.csv (for ground truth)
        test_datasets: Override test datasets
        device: torch device

    Returns:
        List of BenchmarkResult for SelfAttentionBaseline on each test dataset
    """
    print("Running Self-Attention Baseline evaluation...")
    return _evaluate_trained_model(
        model_class="self_attention",
        method_name="SelfAttn-Baseline",
        checkpoint_path=checkpoint_path,
        dataset_tokens_dir=dataset_tokens_dir,
        model_embeddings_dir=model_embeddings_dir,
        csv_path=csv_path,
        test_datasets=test_datasets,
        device=device,
    )


def _evaluate_trained_model(
    model_class: str,
    method_name: str,
    checkpoint_path: str | Path,
    dataset_tokens_dir: str | Path,
    model_embeddings_dir: str | Path,
    csv_path: str | Path,
    test_datasets: list[str] | None,
    device: str,
) -> list[BenchmarkResult]:
    """Internal helper: evaluate a trained model against ground truth."""
    # Get ground truth from Model Spider CSV
    dataset_results = parse_model_spider_csv(csv_path)

    # Get model predictions
    predictions = _load_model_and_predict(
        model_class=model_class,
        checkpoint_path=checkpoint_path,
        dataset_tokens_dir=dataset_tokens_dir,
        model_embeddings_dir=model_embeddings_dir,
        test_datasets=test_datasets,
        device=device,
    )

    all_results: list[BenchmarkResult] = []

    for dataset_name, pred_scores in predictions.items():
        if dataset_name not in dataset_results:
            print(f"  Warning: no ground truth for {dataset_name}, skipping")
            continue

        gt = dataset_results[dataset_name].ground_truth_accuracy
        models = sorted(gt.keys())

        # Only evaluate on models that exist in both gt and predictions
        common_models = [m for m in models if m in pred_scores]
        if len(common_models) < 2:
            print(f"  Warning: fewer than 2 common models for {dataset_name}, skipping")
            continue

        gt_filtered = {m: gt[m] for m in common_models}
        pred_filtered = {m: pred_scores[m] for m in common_models}

        true_ranks, pred_ranks = compute_ranking_from_scores(
            gt_filtered, pred_filtered, higher_is_better=True
        )

        gt_values = [gt_filtered[m] for m in common_models]
        pred_values = [pred_filtered[m] for m in common_models]

        try:
            tau_w, _ = stats.weightedtau(np.array(gt_values), np.array(pred_values))
        except Exception:
            tau_w = 0.0

        try:
            tau_b, _ = stats.kendalltau(np.array(gt_values), np.array(pred_values))
        except Exception:
            tau_b = 0.0

        try:
            rho, _ = stats.spearmanr(np.array(gt_values), np.array(pred_values))
        except Exception:
            rho = 0.0

        ndcg = _calculate_ndcg(np.array(true_ranks), np.array(pred_ranks))
        p_at_1 = compute_precision_at_k(true_ranks, pred_ranks, k=1)
        p_at_3 = compute_precision_at_k(true_ranks, pred_ranks, k=3)
        rel_top1 = compute_relative_top1(gt_filtered, pred_ranks, common_models)

        result = BenchmarkResult(
            dataset_name=dataset_name,
            method_name=method_name,
            tau_w=tau_w,
            tau_b=tau_b,
            spearman_rho=rho,
            ndcg=ndcg,
            precision_at_1=p_at_1,
            precision_at_3=p_at_3,
            relative_top1=rel_top1,
        )
        all_results.append(result)

        print(
            f"  {dataset_name}: τ_w={tau_w:.4f}, P@1={p_at_1:.0f}, "
            f"P@3={p_at_3:.0f}, RelTop1={rel_top1:.4f}"
        )

    if all_results:
        avg_tau_w = np.mean([r.tau_w for r in all_results])
        avg_p1 = np.mean([r.precision_at_1 for r in all_results])
        print(f"\n  {method_name} Average: τ_w={avg_tau_w:.4f}, P@1={avg_p1:.2f}")

    return all_results

