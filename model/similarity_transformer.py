"""Transformer model for scoring model/dataset similarity."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BertConfig, BertModel

from config import ConfigParser, SimilarityModelConfig


class SimilarityTransformerModel(nn.Module):
    """Predicts best-performing models given model and dataset embeddings."""

    def __init__(self) -> None:
        super().__init__()

        self.config = ConfigParser.get(SimilarityModelConfig)

        self.embedding_dim = self.config.embedding_dim
        self.hidden_dim = self.config.hidden_dim
        self.expected_num_models = int(self.config.num_models)
        self._warned_model_count_mismatch = False

        dropout_rate = min(float(self.config.dropout), 0.1)
        attention_dropout = min(float(self.config.attention_dropout), 0.1)
        classifier_dropout = min(float(self.config.classifier_dropout), 0.1)

        self.model_input_norm = nn.LayerNorm(self.embedding_dim)
        self.dataset_input_norm = nn.LayerNorm(self.embedding_dim)
        self.model_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.dataset_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.model_hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.dataset_hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.token_dropout = nn.Dropout(dropout_rate)

        intermediate_dim = self.config.intermediate_dim or self.hidden_dim * 4

        if self.config.use_pretrained and self.config.pretrained_model_name:
            base_config = AutoConfig.from_pretrained(self.config.pretrained_model_name)
            base_config.hidden_size = self.hidden_dim
            base_config.num_hidden_layers = self.config.transformer_layers
            base_config.num_attention_heads = self.config.transformer_heads
            base_config.intermediate_size = intermediate_dim
            base_config.hidden_dropout_prob = dropout_rate
            base_config.attention_probs_dropout_prob = attention_dropout
            self.transformer = AutoModel.from_config(base_config)
        else:
            bert_config = BertConfig(
                hidden_size=self.hidden_dim,
                num_hidden_layers=self.config.transformer_layers,
                num_attention_heads=self.config.transformer_heads,
                intermediate_size=intermediate_dim,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=attention_dropout,
                vocab_size=1,
                type_vocab_size=1,
                max_position_embeddings=self.config.max_position_embeddings,
            )
            self.transformer = BertModel(bert_config)

        self.dataset_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.model_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        temperature_init = max(float(self.config.temperature_init), 1e-6)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        self.temperature_min = float(self.config.temperature_min)
        self.temperature_max = float(self.config.temperature_max)

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

        model_tokens = self.model_input_norm(model_tokens)
        model_tokens = F.normalize(model_tokens, dim=-1)
        model_hidden = self.model_hidden_norm(self.model_proj(model_tokens))
        model_hidden = self.token_dropout(model_hidden)

        dataset_tokens = self.dataset_input_norm(dataset_tokens)
        dataset_tokens = F.normalize(dataset_tokens, dim=-1)
        dataset_hidden = self.dataset_hidden_norm(self.dataset_proj(dataset_tokens))
        dataset_hidden = self.token_dropout(dataset_hidden)

        expanded_models = model_hidden.unsqueeze(0).expand(batch_size, -1, -1)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        transformer_inputs = torch.cat([cls_token, expanded_models, dataset_hidden], dim=1)
        attention_mask = torch.ones(transformer_inputs.size()[:2], dtype=torch.long, device=device)

        outputs = self.transformer(inputs_embeds=transformer_inputs, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        num_models = model_tokens.size(0)
        if (
            self.expected_num_models
            and num_models != self.expected_num_models
            and not self._warned_model_count_mismatch
        ):
            logging.getLogger(__name__).warning(
                "Model received %d candidate embeddings but config expects %d. Proceeding with dynamic sizing.",
                num_models,
                self.expected_num_models,
            )
            self._warned_model_count_mismatch = True
        model_context = sequence_output[:, 1 : 1 + num_models, :]

        dataset_repr = self.dataset_head(cls_output)
        dataset_repr = F.normalize(dataset_repr, dim=-1)

        model_repr = self.model_head(model_context)
        model_repr = self.output_norm(model_repr)
        model_repr = F.normalize(model_repr, dim=-1)

        similarities = torch.einsum("bd,bmd->bm", dataset_repr, model_repr)

        temperature = torch.exp(self.log_temperature).clamp(self.temperature_min, self.temperature_max)
        logits = similarities / temperature

        return {
            "logits": logits,
            "similarities": similarities,
            "dataset_repr": dataset_repr,
            "model_repr": model_repr,
            "temperature": temperature,
        }

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.cls_token = nn.Parameter(self.cls_token.to(*args, **kwargs))
        self.log_temperature = nn.Parameter(self.log_temperature.to(*args, **kwargs))
        return self
