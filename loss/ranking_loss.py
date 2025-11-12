"""Loss utilities for model ranking tasks."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def ranking_loss(
    pred_scores: torch.Tensor,
    true_ranks: torch.Tensor,
    reverse_order: bool = True,
    temperature: float = 1.0
) -> torch.Tensor:
    """Compute a listwise ranking loss inspired by Model Spider.

    Args:
        pred_scores: Predicted relevance scores for each candidate (shape: [M]).
        true_ranks: Ground-truth ranks where 1 denotes the best item by default.
        reverse_order: When True, larger rank values are treated as better items.
        temperature: Temperature for scaling logits. Values > 1.0 make distribution more uniform,
                     helping with initial gradient flow. Default 1.0 (no scaling).

    Returns:
        A scalar tensor representing the listwise ranking loss.
    """
    if type(pred_scores) is not torch.Tensor or type(true_ranks) is not torch.Tensor:
        raise TypeError("pred_scores and true_ranks must be torch.Tensor types.")
    # if pred_scores.dtype != torch.float32 or true_ranks.dtype != torch.float32:
    #     raise ValueError(f"pred_scores and true_ranks must be float32; got {pred_scores.dtype} and {true_ranks.dtype}")
    if pred_scores.ndim > 2 or true_ranks.ndim > 2:
        raise ValueError(f"pred_scores and true_ranks must be 1-D or 2-D; got shapes {tuple(pred_scores.shape)} and {tuple(true_ranks.shape)}")
    if pred_scores.shape[0] != true_ranks.shape[0]:
        raise ValueError("pred_scores and true_ranks must have the same length.")

    if pred_scores.ndim == 1:
        pred_scores = pred_scores.unsqueeze(0)
    if true_ranks.ndim == 1:
        true_ranks = true_ranks.unsqueeze(0)

    # true_ranks contains sorted indices: [best_idx, second_best_idx, ...]
    # Gather predictions in ground truth ranking order
    ordered_scores = torch.gather(pred_scores, dim=1, index=true_ranks.long())

    # Apply temperature scaling (divide scores by temperature)
    # Higher temperature (>1) makes distribution more uniform, improving initial gradients
    if temperature != 1.0:
        ordered_scores = ordered_scores / temperature

    loss_terms = []
    total_candidates = ordered_scores.shape[-1]

    # Plackett-Luce loss: at each position m, compute log P(item_m | remaining items)
    for m in range(total_candidates):
        numerator = ordered_scores[:, m]
        denominator = torch.logsumexp(ordered_scores[:, m:], dim=1)
        loss_terms.append(denominator - numerator)

    return torch.stack(loss_terms).sum()


def pairwise_ranking_loss(
    pred_scores: torch.Tensor,
    true_ranks: torch.Tensor,
    reverse_order: bool = True,
    temperature: float = 1.0
) -> torch.Tensor:
    """Compute a pairwise ranking loss using sequential classification.

    This loss formulation treats ranking as a sequence of classification problems:
    at each position k in the ranking, predict which item should come next from
    the remaining candidates. This is inspired by listwise learning-to-rank methods
    like ListMLE.

    Args:
        pred_scores: Predicted relevance scores (shape: [B, N] or [N]).
        true_ranks: Sorted indices [best_idx, second_best_idx, ...] (shape: [B, N] or [N]).
                    Position 0 contains the index of the best model, position 1 the second-best, etc.
        reverse_order: Deprecated parameter, kept for API compatibility. true_ranks should already be sorted.
        temperature: Temperature for scaling logits. Values > 1.0 make distribution more uniform,
                     helping with initial gradient flow. Default 1.0 (no scaling).

    Returns:
        A scalar tensor representing the pairwise ranking loss.

    Example:
        >>> pred_scores = torch.tensor([[2.5, 1.0, 3.0], [1.5, 2.0, 0.5]])  # [B=2, N=3]
        >>> true_ranks = torch.tensor([[2, 0, 1], [1, 2, 0]])  # sorted indices (pos 0 = best model)
        >>> loss = pairwise_ranking_loss(pred_scores, true_ranks)
    """
    if type(pred_scores) is not torch.Tensor or type(true_ranks) is not torch.Tensor:
        raise TypeError("pred_scores and true_ranks must be torch.Tensor types.")
    if pred_scores.ndim > 2 or true_ranks.ndim > 2:
        raise ValueError(f"pred_scores and true_ranks must be 1-D or 2-D; got shapes {tuple(pred_scores.shape)} and {tuple(true_ranks.shape)}")
    if pred_scores.shape != true_ranks.shape:
        raise ValueError(f"pred_scores and true_ranks must have the same shape; got {pred_scores.shape} and {true_ranks.shape}")

    # Ensure 2-D tensors for batch processing
    squeeze_output = False
    if pred_scores.ndim == 1:
        pred_scores = pred_scores.unsqueeze(0)
        true_ranks = true_ranks.unsqueeze(0)
        squeeze_output = True

    B, N = pred_scores.shape

    # true_ranks already contains sorted indices: [best_idx, second_best_idx, ...]
    # No need to argsort - use them directly
    ordering = true_ranks.long()  # [B, N]

    # Create batch indices for advanced indexing
    batch_idx = torch.arange(B, device=pred_scores.device).unsqueeze(1)  # [B, 1]

    total_loss = torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)

    # For each position in the ranking (except the last)
    # At position k: predict which of the remaining N-k items should come next
    for k in range(N - 1):
        # Get indices of remaining items at position k
        remaining_indices = ordering[:, k:]  # [B, N-k]

        # Get scores for remaining items
        # remaining_scores[b, i] = score of the i-th remaining item in batch b
        remaining_scores = pred_scores[batch_idx, remaining_indices]  # [B, N-k]

        # Apply temperature scaling
        if temperature != 1.0:
            remaining_scores = remaining_scores / temperature

        # Target: the first item in remaining_indices is the correct one (index 0)
        targets = torch.zeros(B, dtype=torch.long, device=pred_scores.device)

        # Compute cross-entropy loss for this classification problem
        log_probs = torch.log_softmax(remaining_scores, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, targets, reduction='sum')
        total_loss = total_loss + loss

    # Normalize by batch size
    return total_loss / B

