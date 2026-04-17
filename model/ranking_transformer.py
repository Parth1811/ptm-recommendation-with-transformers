from __future__ import annotations

import torch
from beautilog import logger
from torch import nn

from config import RankingCrossAttentionTransformerConfig, ConfigParser


class RankingCrossAttentionTransformer(nn.Module):
    """Cross-attention transformer for model-dataset ranking.

    Uses dataset_tokens as source (encoder input) and model_tokens as target (decoder input).
    Outputs logits for ranking the models.
    """

    def __init__(self):
        super().__init__()
        self.config = ConfigParser().get(RankingCrossAttentionTransformerConfig)
        self.d_model = self.config.d_model
        self.num_models = self.config.num_models

        # Transformer with cross-attention
        self.transformer = nn.Transformer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )

        # Output projection to scalar scores per model
        self.output_projection = nn.Linear(self.config.d_model, 1)

    def forward(
        self,
        dataset_tokens: torch.Tensor,
        model_tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through cross-attention transformer.

        Args:
            dataset_tokens: Shape (batch_size, seq_len, d_model) - dataset features
            model_tokens: Shape (batch_size, num_models, d_model) - model embeddings
            src_key_padding_mask: Shape (batch_size, seq_len) - True for padded positions

        Returns:
            logits: Shape (batch_size, num_models) - ranking scores
        """
        # Pass through transformer with cross-attention
        # dataset_tokens -> encoder (source/memory)
        # model_tokens -> decoder (target, attends to encoder via cross-attention)
        output = self.transformer(
            src=dataset_tokens,
            tgt=model_tokens,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project to scalar scores: (batch_size, num_models, d_model) -> (batch_size, num_models)
        logits = self.output_projection(output).squeeze(-1)
        return logits
