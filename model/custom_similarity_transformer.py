"""Cross-attention based model selector using PyTorch MultiheadAttention."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from config import ConfigParser, CustomSimilarityTransformerConfig


logger = logging.getLogger(__name__)


class CustomSimilarityTransformer(nn.Module):
    """Cross-attention transformer for ranking models given dataset features.

    This module implements a learnable cross-attention mechanism between model tokens
    and dataset tokens. It learns to attend over model embeddings based on dataset
    features, producing a probability distribution over models.

    Architecture:
        - Stacked cross-attention layers where:
            - Query (Q) = dataset_tokens
            - Key (K) = model_tokens
            - Value (V) = model_tokens
        - Multi-head attention with configurable heads
        - Layer normalization and optional dropout between layers
        - Final softmax over model dimension to produce probability distribution

    Input/Output Specification:
        Input:
            - model_tokens: Tensor of shape (B, N, D) - N model tokens with embedding dim D
            - dataset_tokens: Tensor of shape (B, M, D) - M dataset tokens with embedding dim D
        Output:
            - probs: Tensor of shape (B, N) - probability distribution over N models (sums to 1)
    """

    def __init__(self, config: CustomSimilarityTransformerConfig | None = None) -> None:
        """Initialize the CustomSimilarityTransformer.

        Args:
            config: Configuration object. If None, loads from config.ini via ConfigParser.
                   All configuration is externalized to ensure consistency across
                   different training/evaluation scenarios.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()

        # Load configuration if not provided
        ConfigParser.load()
        self.config = config or ConfigParser.get(CustomSimilarityTransformerConfig)

        # Store configuration parameters
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers
        self.dropout = self.config.dropout
        self.batch_first = self.config.batch_first

        # Validate configuration
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # Build cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=self.batch_first
            )
            for _ in range(self.num_layers)
        ])

        # Layer normalization for each cross-attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in range(self.num_layers)
        ])

        # Optional dropout for regularization
        self.dropout_layer = nn.Dropout(self.dropout)

        # Initialize parameters with standard initialization
        self._reset_parameters()

        logger.info(
            f"Initialized CustomSimilarityTransformer: "
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}"
        )

    def _reset_parameters(self) -> None:
        """Initialize all learnable parameters using Xavier uniform initialization.

        This is the default initialization for MultiheadAttention layers in PyTorch,
        ensuring stable gradient flow during early training iterations.
        """
        for layer in self.cross_attention_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    def forward(
        self,
        model_tokens: torch.Tensor,
        dataset_tokens: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-attention scores and probability distribution over models.

        The forward pass implements cross-attention where:
        - Dataset tokens serve as queries (what we want to attend to)
        - Model tokens serve as keys and values (what we attend over)

        This allows the model to learn which models are most relevant for a given dataset.

        Args:
            model_tokens: Tensor of shape (B, N, D) where:
                - B: batch size (or 1 for fixed set of model embeddings)
                - N: number of candidate models
                - D: embedding dimension (must equal self.embed_dim)
            dataset_tokens: Tensor of shape (B, M, D) where:
                - B: batch size (matches model_tokens if B > 1)
                - M: number of dataset tokens (variable length is OK)
                - D: embedding dimension (must equal self.embed_dim)
            return_attention_weights: If True, also return raw attention weights
                before softmax normalization. Useful for interpretability and debugging.

        Returns:
            If return_attention_weights is False:
                probs: Tensor of shape (B, N) with values in [0, 1] that sum to 1
                       along the model dimension (N). Represents probability that
                       each model is best for the dataset.

            If return_attention_weights is True:
                tuple of (probs, attention_weights) where:
                - probs: Tensor of shape (B, N)
                - attention_weights: Tensor of shape (B, N) with pre-softmax attention scores

        Raises:
            ValueError: If input shapes are invalid or dimension mismatches occur.
            RuntimeError: If device mismatch between inputs and parameters.

        Example:
            >>> model_tokens = torch.randn(4, 30, 512)  # 4 batches, 30 models, 512-dim
            >>> dataset_tokens = torch.randn(4, 100, 512)  # 4 batches, 100 dataset tokens
            >>> model = CustomSimilarityTransformer()
            >>> probs = model(model_tokens, dataset_tokens)
            >>> print(probs.shape)  # Output: torch.Size([4, 30])
            >>> print(probs[0].sum().item())  # Output: ~1.0 (probability distribution)
        """
        # Input validation
        if model_tokens.dim() != 3:
            raise ValueError(
                f"model_tokens must have 3 dimensions [B, N, D], "
                f"got shape {model_tokens.shape}"
            )
        if dataset_tokens.dim() != 3:
            raise ValueError(
                f"dataset_tokens must have 3 dimensions [B, M, D], "
                f"got shape {dataset_tokens.shape}"
            )

        batch_size_model, num_models, embed_dim_model = model_tokens.shape
        batch_size_dataset, num_dataset_tokens, embed_dim_dataset = dataset_tokens.shape

        # Validate embedding dimensions
        if embed_dim_model != self.embed_dim:
            raise ValueError(
                f"model_tokens embedding dimension {embed_dim_model} "
                f"does not match expected dimension {self.embed_dim}"
            )
        if embed_dim_dataset != self.embed_dim:
            raise ValueError(
                f"dataset_tokens embedding dimension {embed_dim_dataset} "
                f"does not match expected dimension {self.embed_dim}"
            )

        # Ensure both tensors are on the same device
        device = next(self.parameters()).device
        if model_tokens.device != device:
            model_tokens = model_tokens.to(device)
        if dataset_tokens.device != device:
            dataset_tokens = dataset_tokens.to(device)

        # Handle batch size: if model_tokens has batch_size=1, expand to match dataset batch
        if batch_size_model == 1 and batch_size_dataset > 1:
            model_tokens = model_tokens.expand(batch_size_dataset, -1, -1)
            batch_size_model = batch_size_dataset
        elif batch_size_model != batch_size_dataset:
            raise ValueError(
                f"Batch sizes must match or model_tokens must have batch_size=1. "
                f"Got model_tokens batch_size={batch_size_model}, "
                f"dataset_tokens batch_size={batch_size_dataset}"
            )

        # Apply stacked cross-attention layers
        x = dataset_tokens  # (B, M, D) - queries
        attention_output = None

        for i, (attn_layer, norm_layer) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms)
        ):
            # Cross-attention: Q from dataset_tokens, K,V from model_tokens
            # Returns: (output, attention_weights)
            # output shape: (B, M, D)
            # attention_weights shape: (B * num_heads, M, N)
            attn_output, attn_weights = attn_layer(
                query=x,
                key=model_tokens,
                value=model_tokens,
                need_weights=True,
                average_attn_weights=True,  # Average over heads: (B, M, N)
            )

            # Residual connection and layer normalization
            x = norm_layer(x + attn_output)

            # Apply dropout for regularization
            x = self.dropout_layer(x)

            # Store attention weights from last layer for potential return
            if i == self.num_layers - 1:
                attention_output = attn_weights  # (B, M, N)

        # Aggregate dataset token representations via mean pooling
        # This collapses the sequence dimension (M) into a single representation
        # allowing dataset features to collectively vote on model preferences
        pooled_dataset_repr = x.mean(dim=1)  # (B, D)

        # Compute final attention scores via dot product with models
        # This is a learned similarity measure between pooled dataset and each model
        final_scores = torch.matmul(
            pooled_dataset_repr.unsqueeze(1),  # (B, 1, D)
            model_tokens.transpose(-2, -1),     # (B, D, N)
        )  # (B, 1, N)

        # Squeeze the middle dimension and apply softmax for probability distribution
        final_scores = final_scores.squeeze(1)  # (B, N)
        probs = F.softmax(final_scores, dim=-1)  # (B, N) summing to 1 over models

        if return_attention_weights:
            return probs, final_scores
        return probs

    def to(self, *args, **kwargs) -> CustomSimilarityTransformer:
        """Move model to specified device/dtype.

        Args:
            *args: Positional arguments for nn.Module.to()
            **kwargs: Keyword arguments for nn.Module.to()

        Returns:
            self for method chaining
        """
        super().to(*args, **kwargs)
        return self


# Example usage and validation script
if __name__ == "__main__":
    # Create example tensors
    batch_size = 4
    num_models = 30
    num_dataset_tokens = 100
    embed_dim = 512

    model_tokens = torch.randn(batch_size, num_models, embed_dim)
    dataset_tokens = torch.randn(batch_size, num_dataset_tokens, embed_dim)

    # Initialize model
    model = CustomSimilarityTransformer()
    model.eval()

    # Forward pass
    with torch.no_grad():
        probs = model(model_tokens, dataset_tokens)
        probs_with_scores, scores = model(
            model_tokens, dataset_tokens, return_attention_weights=True
        )

    # Validate outputs
    print("CustomSimilarityTransformer Example Usage")
    print("=" * 50)
    print(f"Input Shapes:")
    print(f"  model_tokens: {model_tokens.shape}")
    print(f"  dataset_tokens: {dataset_tokens.shape}")
    print(f"\nOutput Shapes:")
    print(f"  probs: {probs.shape}")
    print(f"  scores: {scores.shape}")
    print(f"\nValidation:")
    print(f"  probs min/max: {probs.min().item():.4f} / {probs.max().item():.4f}")
    print(f"  probs sum per batch (should be ~1.0):")
    for b in range(min(2, batch_size)):
        print(f"    batch {b}: {probs[b].sum().item():.6f}")
    print(f"  scores min/max: {scores.min().item():.4f} / {scores.max().item():.4f}")

    # Test gradient flow
    print(f"\nGradient Flow Test:")
    loss = probs.mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Number of parameters with gradients: {len(grad_norms)}")
    print(f"  Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
    print(f"  Gradient range: {min(grad_norms):.6f} to {max(grad_norms):.6f}")
    print(f"\nTest completed successfully!")
