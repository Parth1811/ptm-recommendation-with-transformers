"""Loss utilities for model ranking tasks."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def ranking_loss(pred_scores: torch.Tensor, true_ranks: torch.Tensor, reverse_order: bool = True) -> torch.Tensor:
    """Compute a listwise ranking loss inspired by Model Spider.

    Args:
        pred_scores: Predicted relevance scores for each candidate (shape: [M]).
        true_ranks: Ground-truth ranks where 1 denotes the best item by default.
        reverse_order: When True, larger rank values are treated as better items.

    Returns:
        A scalar tensor representing the listwise ranking loss.
    """
    if type(pred_scores) is not torch.Tensor or type(true_ranks) is not torch.Tensor:
        raise TypeError("pred_scores and true_ranks must be torch.Tensor types.")
    if pred_scores.dtype != torch.float32 or true_ranks.dtype != torch.float32:
        raise ValueError(f"pred_scores and true_ranks must be float32; got {pred_scores.dtype} and {true_ranks.dtype}")
    if pred_scores.ndim != 1 or true_ranks.ndim != 1:
        raise ValueError(f"pred_scores and true_ranks must be 1-D; got shapes {tuple(pred_scores.shape)} and {tuple(true_ranks.shape)}")
    if pred_scores.shape[0] != true_ranks.shape[0]:
        raise ValueError("pred_scores and true_ranks must have the same length.")

    ordering = torch.argsort(true_ranks, descending=bool(reverse_order))
    ordered_scores = pred_scores[ordering]
    loss_terms = []
    total_candidates = ordered_scores.shape[0]

    for m in range(total_candidates):
        numerator = ordered_scores[m]
        denominator = torch.logsumexp(ordered_scores[m:], dim=0)
        loss_terms.append(denominator - numerator)

    return torch.stack(loss_terms).sum()

