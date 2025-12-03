"""Evaluation script for ranking metrics based on Kendall's rank correlation.

This script implements evaluation metrics from:
- Kendall, M. G. (1938). "A NEW MEASURE OF RANK CORRELATION". Biometrika, 30(1-2), 81-93.
- Weighted Kendall's tau (τ_w) as used in Model Spider (NeurIPS 2023)

Calculates rank correlation metrics between predicted and true model rankings.
The weighted tau gives higher importance to correctly ranking top items, which is
critical for model recommendation tasks.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from beautilog import logger
from scipy import stats

logger.name = "RankingEvaluation"


@dataclass
class RankingMetrics:
    """Container for ranking evaluation metrics."""

    kendall_tau_a: float
    kendall_tau_b: float
    kendall_tau_w: float  # Weighted tau (Model Spider metric)
    spearman_rho: float
    ndcg: float
    average_precision: float
    num_concordant: int
    num_discordant: int
    num_ties: int


def parse_rank_list(rank_str: str) -> list[int]:
    """Parse a rank list from CSV string representation.

    Args:
        rank_str: String representation of a list, e.g., "[0, 1, 2, 3]"

    Returns:
        List of integers representing ranks
    """
    return ast.literal_eval(rank_str)


def calculate_kendall_tau_a(true_ranks: np.ndarray, pred_ranks: np.ndarray) -> tuple[float, int, int]:
    """Calculate Kendall's Tau-a coefficient.

    Kendall's Tau-a is the original measure from the 1938 paper:
    τ_a = (C - D) / (n(n-1)/2)

    where:
    - C = number of concordant pairs
    - D = number of discordant pairs
    - n = number of items

    A concordant pair is where both rankings agree on the relative order.
    A discordant pair is where the rankings disagree.

    Args:
        true_ranks: True ranking (ground truth)
        pred_ranks: Predicted ranking

    Returns:
        Tuple of (tau_a, concordant_count, discordant_count)
    """
    n = len(true_ranks)
    concordant = 0
    discordant = 0

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Check if true ranks agree on order
            true_order = np.sign(true_ranks[i] - true_ranks[j])
            pred_order = np.sign(pred_ranks[i] - pred_ranks[j])

            if true_order * pred_order > 0:
                concordant += 1
            elif true_order * pred_order < 0:
                discordant += 1
            # If product is 0, it's a tie in at least one ranking

    total_pairs = n * (n - 1) // 2
    tau_a = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    return tau_a, concordant, discordant


def calculate_ndcg(true_ranks: np.ndarray, pred_ranks: np.ndarray, k: int | None = None) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    NDCG measures how well the predicted ranking matches the true ranking,
    with higher weight given to correctly ranking items at the top.

    Args:
        true_ranks: True ranking (lower is better, 0 = best)
        pred_ranks: Predicted ranking (lower is better, 0 = best)
        k: Only consider top k items (if None, use all)

    Returns:
        NDCG score between 0 and 1
    """
    if k is None:
        k = len(true_ranks)

    # Convert ranks to relevance scores (inverse of rank, so best rank gets highest score)
    n = len(true_ranks)
    true_relevance = n - true_ranks.astype(float)

    # Get order of predicted ranks
    pred_order = np.argsort(pred_ranks)[:k]

    # Calculate DCG for predicted ranking
    dcg = 0.0
    for i, idx in enumerate(pred_order):
        # Position i+1 in predicted ranking contains item idx
        relevance = true_relevance[idx]
        dcg += relevance / np.log2(i + 2)  # i+2 because positions start at 1

    # Calculate ideal DCG (best possible)
    ideal_order = np.argsort(true_ranks)[:k]
    idcg = 0.0
    for i, idx in enumerate(ideal_order):
        relevance = true_relevance[idx]
        idcg += relevance / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_average_precision(true_ranks: np.ndarray, pred_ranks: np.ndarray, k: int | None = None) -> float:
    """Calculate Average Precision (AP).

    AP measures the average precision at each position where a relevant item appears.

    Args:
        true_ranks: True ranking (lower is better)
        pred_ranks: Predicted ranking (lower is better)
        k: Only consider top k items (if None, use all)

    Returns:
        Average precision score
    """
    if k is None:
        k = len(true_ranks)

    # Items are "relevant" if they're in the top half of true ranking
    threshold = len(true_ranks) // 2
    relevant_items = set(np.where(true_ranks < threshold)[0])

    if len(relevant_items) == 0:
        return 0.0

    # Get predicted order
    pred_order = np.argsort(pred_ranks)[:k]

    # Calculate precision at each relevant item position
    precisions = []
    num_relevant_seen = 0

    for i, idx in enumerate(pred_order):
        if idx in relevant_items:
            num_relevant_seen += 1
            precision = num_relevant_seen / (i + 1)
            precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def evaluate_ranking(true_ranks: list[int], pred_ranks: list[int]) -> RankingMetrics:
    """Evaluate ranking quality using multiple metrics.

    Args:
        true_ranks: Ground truth ranking
        pred_ranks: Predicted ranking

    Returns:
        RankingMetrics object with all computed metrics
    """
    true_array = np.array(true_ranks)
    pred_array = np.array(pred_ranks)

    # Calculate Kendall's Tau variants
    tau_a, concordant, discordant = calculate_kendall_tau_a(true_array, pred_array)

    # Tau-b (accounts for ties)
    tau_b, _ = stats.kendalltau(true_array, pred_array)

    # Weighted Tau (τ_w) - Used in Model Spider paper
    # Gives higher importance to correctly ranking items at the top
    # Weight for exchange between ranks r and s: 1/(r+1) + 1/(s+1)
    try:
        tau_w, _ = stats.weightedtau(true_array, pred_array)
    except Exception:
        tau_w = tau_b  # Fallback to tau-b if weighted tau fails

    # Calculate Spearman's rho for comparison
    spearman_rho, _ = stats.spearmanr(true_array, pred_array)

    # Calculate NDCG
    ndcg = calculate_ndcg(true_array, pred_array)

    # Calculate Average Precision
    avg_precision = calculate_average_precision(true_array, pred_array)

    # Count ties
    num_ties = len(true_ranks) * (len(true_ranks) - 1) // 2 - concordant - discordant

    return RankingMetrics(
        kendall_tau_a=tau_a,
        kendall_tau_b=tau_b,
        kendall_tau_w=tau_w,
        spearman_rho=spearman_rho,
        ndcg=ndcg,
        average_precision=avg_precision,
        num_concordant=concordant,
        num_discordant=discordant,
        num_ties=num_ties,
    )


def load_results(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load transformer results from CSV.

    Args:
        csv_path: Path to transformer_results.csv

    Returns:
        List of result dictionaries
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def main(results_path: str | Path = "artifacts/transformer_results.csv", output_path: str | Path | None = None):
    """Run evaluation on transformer results.

    Args:
        results_path: Path to transformer_results.csv
        output_path: Optional path to save detailed metrics CSV
    """
    results_path = Path(results_path)

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    # Load and evaluate results
    results = load_results(results_path)

    # Group results by dataset name
    from collections import defaultdict
    dataset_metrics: dict[str, list[RankingMetrics]] = defaultdict(list)

    for row in results:
        true_ranks = parse_rank_list(row['true_ranks'])
        pred_ranks = parse_rank_list(row['predicted_ranks'])

        metrics = evaluate_ranking(true_ranks, pred_ranks)
        dataset_metrics[row['dataset_name']].append(metrics)

    # Average metrics for each unique dataset
    averaged_results = []
    for dataset_name, metrics_list in dataset_metrics.items():
        avg_result = {
            'dataset_name': dataset_name,
            'kendall_tau_a': np.mean([m.kendall_tau_a for m in metrics_list]),
            'kendall_tau_b': np.mean([m.kendall_tau_b for m in metrics_list]),
            'kendall_tau_w': np.mean([m.kendall_tau_w for m in metrics_list]),
            'spearman_rho': np.mean([m.spearman_rho for m in metrics_list]),
            'ndcg': np.mean([m.ndcg for m in metrics_list]),
            'average_precision': np.mean([m.average_precision for m in metrics_list]),
            'num_samples': len(metrics_list),
        }
        averaged_results.append(avg_result)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RANKING EVALUATION RESULTS")
    logger.info("=" * 80)

    # Overall aggregate metrics (averaged across unique datasets)
    metrics_arrays = {
        'Kendall Tau-a': [r['kendall_tau_a'] for r in averaged_results],
        'Kendall Tau-b': [r['kendall_tau_b'] for r in averaged_results],
        'Kendall Tau-w': [r['kendall_tau_w'] for r in averaged_results],
        'Spearman Rho': [r['spearman_rho'] for r in averaged_results],
        'NDCG': [r['ndcg'] for r in averaged_results],
        'Average Precision': [r['average_precision'] for r in averaged_results],
    }

    logger.info("\nOVERALL METRICS (n={} unique datasets):".format(len(averaged_results)))
    logger.info("-" * 80)
    for metric_name, values in metrics_arrays.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"{metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Per-dataset breakdown
    logger.info("\n" + "=" * 80)
    logger.info("PER-DATASET METRICS")
    logger.info("=" * 80)

    # Sort by tau_w (primary metric) descending
    sorted_results = sorted(averaged_results, key=lambda x: x['kendall_tau_w'], reverse=True)

    logger.info(f"\n{'Dataset':<20} {'τ_a':>8} {'τ_b':>8} {'τ_w':>8} {'ρ':>8} {'NDCG':>8} {'AP':>8}")
    logger.info("-" * 80)
    for result in sorted_results:
        logger.info(
            f"{result['dataset_name']:<20} "
            f"{result['kendall_tau_a']:>8.4f} "
            f"{result['kendall_tau_b']:>8.4f} "
            f"{result['kendall_tau_w']:>8.4f} "
            f"{result['spearman_rho']:>8.4f} "
            f"{result['ndcg']:>8.4f} "
            f"{result['average_precision']:>8.4f}"
        )

    # Save detailed results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            fieldnames = list(averaged_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(averaged_results)

        logger.info(f"\nDetailed metrics saved to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("METRIC LEGEND")
    logger.info("-" * 80)
    logger.info("τ_a  = Kendall Tau-a (original, no tie adjustment)")
    logger.info("τ_b  = Kendall Tau-b (adjusted for ties)")
    logger.info("τ_w  = Kendall Tau-w (weighted, Model Spider metric)")
    logger.info("       Weight: 1/(r+1) + 1/(s+1) for ranks r,s")
    logger.info("ρ    = Spearman's Rho (Pearson correlation on ranks)")
    logger.info("NDCG = Normalized Discounted Cumulative Gain")
    logger.info("AP   = Average Precision")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        main(results_path, output_path)
    else:
        # Default paths
        main(
            results_path="artifacts/transformer_results.csv",
            output_path="artifacts/ranking_metrics.csv"
        )
