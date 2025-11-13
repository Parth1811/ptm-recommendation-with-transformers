"""Trainer registry."""

from .model_autoencoder_trainer import ModelAutoEncoderTrainer
from .transformer_trainer import TransformerTrainer

TRAINER_REGISTRY = {
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
    "TransformerTrainer": TransformerTrainer,
}

