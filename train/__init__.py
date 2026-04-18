"""Trainer registry."""

from .model_autoencoder_trainer import ModelAutoEncoderTrainer
from .self_attention_baseline_trainer import SelfAttentionBaselineTrainer
from .transformer_trainer import TransformerTrainer

TRAINER_REGISTRY = {
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
    "TransformerTrainer": TransformerTrainer,
    "SelfAttentionBaselineTrainer": SelfAttentionBaselineTrainer,
}
