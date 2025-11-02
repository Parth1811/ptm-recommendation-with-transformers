"""Trainer registry."""

from .model_autoencoder_trainer import ModelAutoEncoderTrainer

TRAINER_REGISTRY = {
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
}

__all__ = ["TRAINER_REGISTRY", "ModelAutoEncoderTrainer"]
