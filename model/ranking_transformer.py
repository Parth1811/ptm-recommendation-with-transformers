"""Cross-attention transformer for model-dataset ranking.

Uses nn.MultiheadAttention directly (no causal masking) so that each model
token can attend to ALL dataset tokens bidirectionally. This is the correct
architecture for ranking — nn.Transformer's decoder applies causal masking
which prevents models from comparing against each other.
"""

import torch
from beautilog import logger
from torch import nn

from config import RankingCrossAttentionTransformerConfig, ConfigParser


class RankingCrossAttentionTransformer(nn.Module):
    """Bidirectional cross-attention transformer for model-dataset ranking.

    Architecture:
        1. Input projections + LayerNorm to align embedding spaces
        2. Self-attention over dataset tokens (encoder)
        3. Cross-attention: model tokens (Q) attend to dataset tokens (K, V)
        4. FFN + residual connections
        5. Linear projection to scalar scores per model
    """

    def __init__(self):
        super().__init__()
        self.config = ConfigParser().get(RankingCrossAttentionTransformerConfig)
        d_model = self.config.d_model
        nhead = self.config.nhead
        dropout = self.config.dropout
        dim_feedforward = self.config.dim_feedforward
        num_encoder_layers = self.config.num_encoder_layers
        num_decoder_layers = self.config.num_decoder_layers

        # Input projections to align different embedding spaces
        self.dataset_projection = nn.Linear(d_model, d_model)
        self.model_projection = nn.Linear(d_model, d_model)
        self.dataset_input_norm = nn.LayerNorm(d_model)
        self.model_input_norm = nn.LayerNorm(d_model)

        # Encoder: self-attention over dataset tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.dataset_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Cross-attention layers: model tokens attend to encoded dataset tokens
        self.cross_attention_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.cross_ffn_layers = nn.ModuleList()
        self.cross_ffn_norms = nn.ModuleList()

        for _ in range(num_decoder_layers):
            self.cross_attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.cross_attn_norms.append(nn.LayerNorm(d_model))
            self.cross_ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.cross_ffn_norms.append(nn.LayerNorm(d_model))

        # Output projection to scalar scores per model
        self.output_projection = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        dataset_tokens: torch.Tensor,
        model_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through bidirectional cross-attention transformer.

        Args:
            dataset_tokens: Shape (batch_size, seq_len, d_model) - dataset features
            model_tokens: Shape (batch_size, num_models, d_model) - model embeddings

        Returns:
            logits: Shape (batch_size, num_models) - ranking scores
        """
        # 1. Project and normalize inputs
        d = self.dataset_input_norm(self.dataset_projection(dataset_tokens))
        m = self.model_input_norm(self.model_projection(model_tokens))

        # 2. Encode dataset tokens with self-attention
        d = self.dataset_encoder(d)

        # 3. Cross-attention: model tokens (Q) attend to dataset tokens (K, V)
        #    NO causal mask — each model can see all dataset tokens
        x = m
        for cross_attn, attn_norm, ffn, ffn_norm in zip(
            self.cross_attention_layers,
            self.cross_attn_norms,
            self.cross_ffn_layers,
            self.cross_ffn_norms,
        ):
            # Cross-attention with residual
            attn_out, _ = cross_attn(query=x, key=d, value=d)
            x = attn_norm(x + attn_out)

            # FFN with residual
            x = ffn_norm(x + ffn(x))

        # 4. Project to scalar scores: (B, N, D) -> (B, N, 1) -> (B, N)
        logits = self.output_projection(x).squeeze(-1)

        return logits
