"""Self-attention baseline model for ablation comparison with Cross-Select.

This implements a self-attention baseline that mimics Model Spider's architecture:
- Concatenates model tokens and dataset tokens into a single sequence
- Applies self-attention (vs. Cross-Select's cross-attention)
- Produces a probability distribution over models

This serves as a controlled ablation (A1) that isolates the effect of
cross-attention vs. self-attention, using the same input representations.

Reference:
    Zhang et al., "Model Spider: Learning to Rank Pre-Trained Models
    Efficiently", NeurIPS 2023.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class SelfAttentionBaseline(nn.Module):
    """Self-attention baseline for model ranking (Model Spider proxy).

    Unlike Cross-Select's cross-attention, this model:
    1. Concatenates [dataset_tokens; model_tokens] into a single sequence
    2. Applies standard self-attention across the full sequence
    3. Extracts model positions from the output
    4. Produces rankings via a linear classification head

    This architecture mirrors Model Spider's core mechanism, enabling
    a direct comparison between self-attention and cross-attention.

    Input/Output:
        Input:
            - model_tokens: (B, N, D) - N model embeddings
            - dataset_tokens: (B, M, D) - M dataset tokens
        Output:
            - probs: (B, N) - probability distribution over N models
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_learnable_model_tokens: bool = False,
        num_models: int = 10,
    ) -> None:
        """Initialize the SelfAttentionBaseline.

        Args:
            embed_dim: Embedding dimension (must match model/dataset token dim)
            num_heads: Number of attention heads
            num_layers: Number of self-attention layers (Model Spider uses 1)
            dropout: Dropout rate
            use_learnable_model_tokens: If True, use learnable tokens instead of
                model embeddings (exactly like Model Spider). If False, use
                autoencoder-derived model embeddings (for fair comparison).
            num_models: Number of models in the zoo (only used if
                use_learnable_model_tokens=True)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_learnable_model_tokens = use_learnable_model_tokens
        self.num_models = num_models

        # Optional learnable model tokens (true Model Spider style)
        if use_learnable_model_tokens:
            self.model_tokens = nn.Parameter(
                torch.randn(1, num_models, embed_dim) * 0.02
            )
            logger.info(
                f"Using learnable model tokens: {num_models} × {embed_dim}"
            )

        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        # Feed-forward layers (one per self-attention layer)
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        # Classification head: project model token outputs to a score
        self.classification_head = nn.Linear(embed_dim, 1)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Type embeddings to distinguish model vs. dataset tokens
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0=dataset, 1=model

        self._reset_parameters()

        logger.info(
            f"Initialized SelfAttentionBaseline: embed_dim={embed_dim}, "
            f"num_heads={num_heads}, num_layers={num_layers}, "
            f"learnable_tokens={use_learnable_model_tokens}"
        )

    def _reset_parameters(self) -> None:
        """Xavier initialization for stable training."""
        for layer in self.self_attention_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)

    def forward(
        self,
        model_tokens: torch.Tensor,
        dataset_tokens: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: concatenate, self-attend, classify.

        Args:
            model_tokens: (B, N, D) model embeddings
                (ignored if use_learnable_model_tokens=True)
            dataset_tokens: (B, M, D) dataset tokens
            return_attention_weights: If True, return (probs, scores)

        Returns:
            probs: (B, N) probability distribution over models
        """
        batch_size = dataset_tokens.shape[0]
        device = dataset_tokens.device

        # Get model tokens
        if self.use_learnable_model_tokens:
            m_tokens = self.model_tokens.expand(batch_size, -1, -1)
        else:
            m_tokens = model_tokens
            if m_tokens.shape[0] == 1 and batch_size > 1:
                m_tokens = m_tokens.expand(batch_size, -1, -1)

        num_models = m_tokens.shape[1]
        num_dataset = dataset_tokens.shape[1]

        # Add type embeddings
        dataset_type_ids = torch.zeros(
            batch_size, num_dataset, dtype=torch.long, device=device
        )
        model_type_ids = torch.ones(
            batch_size, num_models, dtype=torch.long, device=device
        )

        d_tokens = dataset_tokens + self.type_embedding(dataset_type_ids)
        m_tokens_typed = m_tokens + self.type_embedding(model_type_ids)

        # Concatenate: [dataset_tokens; model_tokens] -> (B, M+N, D)
        x = torch.cat([d_tokens, m_tokens_typed], dim=1)

        # Apply self-attention layers
        for attn, norm, ffn, ffn_norm in zip(
            self.self_attention_layers,
            self.layer_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            # Self-attention with residual
            attn_out, _ = attn(query=x, key=x, value=x)
            x = norm(x + attn_out)

            # FFN with residual
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)

        # Extract model token positions (last N tokens)
        model_outputs = x[:, num_dataset:, :]  # (B, N, D)

        # Classification head: score each model
        scores = self.classification_head(model_outputs).squeeze(-1)  # (B, N)

        # Softmax to get probability distribution
        probs = F.softmax(scores, dim=-1)  # (B, N)

        if return_attention_weights:
            return probs, scores
        return probs


# Quick validation
if __name__ == "__main__":
    batch_size = 4
    num_models = 10
    num_dataset_tokens = 50
    embed_dim = 512

    model_tokens = torch.randn(batch_size, num_models, embed_dim)
    dataset_tokens = torch.randn(batch_size, num_dataset_tokens, embed_dim)

    # Test with autoencoder-derived model embeddings
    model_ae = SelfAttentionBaseline(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=1,
        use_learnable_model_tokens=False,
    )
    model_ae.eval()
    with torch.no_grad():
        probs = model_ae(model_tokens, dataset_tokens)
    print(f"SelfAttentionBaseline (AE tokens)")
    print(f"  Input: model={model_tokens.shape}, dataset={dataset_tokens.shape}")
    print(f"  Output: probs={probs.shape}, sum={probs[0].sum():.4f}")
    print(f"  Params: {sum(p.numel() for p in model_ae.parameters()):,}")

    # Test with learnable model tokens (true Model Spider)
    model_lt = SelfAttentionBaseline(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=1,
        use_learnable_model_tokens=True,
        num_models=num_models,
    )
    model_lt.eval()
    with torch.no_grad():
        probs_lt = model_lt(model_tokens, dataset_tokens)
    print(f"\nSelfAttentionBaseline (Learnable tokens)")
    print(f"  Output: probs={probs_lt.shape}, sum={probs_lt[0].sum():.4f}")
    print(f"  Params: {sum(p.numel() for p in model_lt.parameters()):,}")

    # Gradient flow test
    loss = probs.mean()
    loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in model_ae.parameters() if p.grad is not None
    ]
    print(f"\nGradient flow: {len(grad_norms)} params have gradients")
    print(f"  Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
    print("Test passed!")
