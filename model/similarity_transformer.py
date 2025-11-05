"""Transformer model for scoring model/dataset similarity."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, BertConfig, BertModel

from config import ConfigParser, SimilarityModelConfig


class SimilarityTransformerModel(nn.Module):
    """Predicts best-performing models given model and dataset embeddings."""

    def __init__(self) -> None:
        super().__init__()

        self.config = ConfigParser.get(SimilarityModelConfig)

        self.embedding_dim = self.config.embedding_dim
        self.hidden_dim = self.config.hidden_dim
        self.num_models = self.config.num_models

        self.model_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.dataset_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.token_dropout = nn.Dropout(self.config.dropout)

        intermediate_dim = self.config.intermediate_dim or self.hidden_dim * 4

        if self.config.use_pretrained and self.config.pretrained_model_name:
            base_config = AutoConfig.from_pretrained(self.config.pretrained_model_name)
            base_config.hidden_size = self.hidden_dim
            base_config.num_hidden_layers = self.config.transformer_layers
            base_config.num_attention_heads = self.config.transformer_heads
            base_config.intermediate_size = intermediate_dim
            base_config.hidden_dropout_prob = self.config.dropout
            base_config.attention_probs_dropout_prob = self.config.attention_dropout
            self.transformer = AutoModel.from_config(base_config)
        else:
            bert_config = BertConfig(
                hidden_size=self.hidden_dim,
                num_hidden_layers=self.config.transformer_layers,
                num_attention_heads=self.config.transformer_heads,
                intermediate_size=intermediate_dim,
                hidden_dropout_prob=self.config.dropout,
                attention_probs_dropout_prob=self.config.attention_dropout,
                vocab_size=1,
                type_vocab_size=1,
                max_position_embeddings=self.config.max_position_embeddings,
            )
            self.transformer = BertModel(bert_config)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.hidden_dim, self.num_models),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        model_tokens: torch.Tensor,
        dataset_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if model_tokens.dim() != 2:
            raise ValueError("model_tokens must have shape [num_models, embedding_dim]")
        if dataset_tokens.dim() != 3:
            raise ValueError("dataset_tokens must have shape [batch, num_tokens, embedding_dim]")

        device = self.cls_token.device
        if model_tokens.device != device:
            model_tokens = model_tokens.to(device)
        if dataset_tokens.device != device:
            dataset_tokens = dataset_tokens.to(device)

        batch_size = dataset_tokens.size(0)
        model_tokens = torch.repeat_interleave(model_tokens.unsqueeze(0), batch_size, dim=0)

        model_hidden = self.token_dropout(self.model_proj(model_tokens))
        dataset_hidden = self.token_dropout(self.dataset_proj(dataset_tokens))

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        transformer_inputs = torch.cat([cls_token, model_hidden, dataset_hidden], dim=1)
        attention_mask = torch.ones(transformer_inputs.size()[:2], dtype=torch.long, device=device)

        outputs = self.transformer(inputs_embeds=transformer_inputs, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.cls_token = nn.Parameter(self.cls_token.to(*args, **kwargs))
        return self
