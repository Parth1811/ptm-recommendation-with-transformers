"""Evaluate trained Cross-Select and SelfAttentionBaseline models.

Loads a trained checkpoint, runs inference on each test dataset using
pre-extracted features, and computes the same ranking metrics as the
Model Spider baselines for direct comparison.

Usage:
    from evaluate_models import run_cross_select_evaluation
    results = run_cross_select_evaluation("artifacts/models/.../checkpoint.pt")

See also:
    evaluate_benchmark.py  — CSV parsing, metric helpers, baseline evaluation
    baselines/             — Transferability estimation methods
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from evaluate_benchmark import (
    BenchmarkResult,
    TEST_DATASETS,
    _calculate_ndcg,
    _compute_benchmark_result,
    compute_precision_at_k,
    compute_ranking_from_scores,
    compute_relative_top1,
    parse_model_spider_csv,
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_model_and_predict(
    model_class: str,
    checkpoint_path: Path,
    dataset_tokens_dir: Path,
    model_embeddings_dir: Path,
    test_datasets: list[str],
    device: str,
) -> dict[str, dict[str, float]]:
    """Load a trained model and compute predicted ranking scores per dataset.

    Args:
        model_class: ``"cross_select"`` or ``"self_attention"``
        checkpoint_path: Path to .pt checkpoint
        dataset_tokens_dir: Root dir containing dataset token shards (npz)
        model_embeddings_dir: Root dir containing model embedding .npz files
        test_datasets: List of test dataset names
        device: torch device string

    Returns:
        ``{dataset_name: {model_name: predicted_score}}``
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_class == "cross_select":
        from model import CustomSimilarityTransformer
        model = CustomSimilarityTransformer()
    elif model_class == "self_attention":
        from model import SelfAttentionBaseline
        model = SelfAttentionBaseline()
    else:
        raise ValueError(f"Unknown model_class: {model_class!r}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load all model embeddings
    model_files = sorted(f for f in model_embeddings_dir.rglob("*.npz") if f.is_file())
    model_names: list[str] = []
    embedding_tensors: list[object] = []  # torch.Tensor at runtime
    for fpath in model_files:
        with np.load(fpath) as archive:
            emb = archive["embedding"].flatten().astype("float32")
        embedding_tensors.append(torch.tensor(emb))
        model_names.append(fpath.stem)

    # (1, num_models, D_model)
    model_tokens = torch.stack(embedding_tensors, dim=0).unsqueeze(0).to(device)

    all_predictions: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for dataset_name in test_datasets:
            dataset_dir = dataset_tokens_dir / dataset_name
            if not dataset_dir.exists():
                print(f"  Warning: dataset tokens not found for {dataset_name}, skipping")
                continue

            # Prefer test split, fall back to validation then train
            shard_files: list[Path] = []
            for split in ("test", "validation", "train"):
                split_dir = dataset_dir / split
                if split_dir.exists():
                    shard_files = sorted(split_dir.glob("*.npz"))
                    if shard_files:
                        break

            if not shard_files:
                print(f"  Warning: no shard files for {dataset_name}, skipping")
                continue

            batch_scores: list[object] = []  # torch.Tensor at runtime
            for shard_path in shard_files:
                with np.load(shard_path, allow_pickle=True) as archive:
                    features = archive["features"]  # (batches, num_classes, dim)

                dataset_tokens = torch.tensor(features, dtype=torch.float32).to(device)
                batch_size = dataset_tokens.shape[0]
                m_tokens = model_tokens.expand(batch_size, -1, -1)

                if model_class == "cross_select":
                    scores = model(m_tokens, dataset_tokens)        # (B, num_models)
                else:  # self_attention
                    _, scores = model(
                        m_tokens, dataset_tokens, return_attention_weights=True
                    )

                batch_scores.append(scores.cpu())

            avg_scores = torch.cat(batch_scores, dim=0).mean(dim=0)  # (num_models,)
            all_predictions[dataset_name] = {
                model_names[i]: avg_scores[i].item()
                for i in range(len(model_names))
            }

    return all_predictions


# ---------------------------------------------------------------------------
# Shared evaluation loop
# ---------------------------------------------------------------------------

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
    """Core evaluation loop shared by Cross-Select and SelfAttn baselines."""
    checkpoint_path = Path(checkpoint_path)
    dataset_tokens_dir = Path(dataset_tokens_dir)
    model_embeddings_dir = Path(model_embeddings_dir)
    test_datasets = test_datasets or TEST_DATASETS

    dataset_results = parse_model_spider_csv(csv_path)

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
        common_models = sorted(m for m in gt if m in pred_scores)

        if len(common_models) < 2:
            print(f"  Warning: fewer than 2 common models for {dataset_name}, skipping")
            continue

        gt_filtered = {m: gt[m] for m in common_models}
        pred_filtered = {m: pred_scores[m] for m in common_models}

        result = _compute_benchmark_result(
            dataset_name=dataset_name,
            method_name=method_name,
            gt=gt_filtered,
            scores=pred_filtered,
            higher_is_better=True,
        )
        all_results.append(result)

        print(
            f"  {dataset_name}: τ_w={result.tau_w:.4f}, "
            f"P@1={result.precision_at_1:.0f}, "
            f"P@3={result.precision_at_3:.0f}, "
            f"RelTop1={result.relative_top1:.4f}"
        )

    if all_results:
        avg_tau_w = float(np.mean([r.tau_w for r in all_results]))
        avg_p1 = float(np.mean([r.precision_at_1 for r in all_results]))
        print(f"\n  {method_name} Average: τ_w={avg_tau_w:.4f}, P@1={avg_p1:.2f}")

    return all_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cross_select_evaluation(
    checkpoint_path: str | Path,
    dataset_tokens_dir: str | Path = "artifacts/extracted/datasets",
    model_embeddings_dir: str | Path = "artifacts/extracted/model_embeddings",
    csv_path: str | Path = "constants/model_spider_baseline_results.csv",
    test_datasets: list[str] | None = None,
    device: str = "cuda",
) -> list[BenchmarkResult]:
    """Evaluate a trained Cross-Select model on the Model Spider benchmark.

    Args:
        checkpoint_path: Path to trained CustomSimilarityTransformer .pt file
        dataset_tokens_dir: Root dir for dataset token shards
        model_embeddings_dir: Root dir for model embedding .npz files
        csv_path: Path to baseline_results.csv (provides ground truth)
        test_datasets: Override test datasets (default: TEST_DATASETS)
        device: torch device string

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
        csv_path: Path to baseline_results.csv (provides ground truth)
        test_datasets: Override test datasets (default: TEST_DATASETS)
        device: torch device string

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
