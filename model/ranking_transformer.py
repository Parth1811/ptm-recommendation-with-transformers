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
        model_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through cross-attention transformer.

        Args:
            dataset_tokens: Shape (batch_size, seq_len, d_model) - dataset features
            model_tokens: Shape (batch_size, num_models, d_model) - model embeddings

        Returns:
            logits: Shape (batch_size, num_models) - ranking scores
        """
        # Transformer expects (batch_size, seq_len, d_model) for both inputs
        # dataset_tokens -> encoder (source/memory)
        # model_tokens -> decoder (target)

        # Pass through transformer with cross-attention
        # output shape: (batch_size, num_models, d_model)
        output = self.transformer(
            src=dataset_tokens,  # Encoder input
            tgt=model_tokens,    # Decoder input (attends to encoder via cross-attention)
        )

        # Project to scalar scores: (batch_size, num_models, d_model) -> (batch_size, num_models, 1)
        logits = self.output_projection(output)

        # Squeeze last dimension: (batch_size, num_models, 1) -> (batch_size, num_models)
        logits = logits.squeeze(-1)
        # Return raw logits - ranking loss will handle score transformation via logsumexp

        return logits
