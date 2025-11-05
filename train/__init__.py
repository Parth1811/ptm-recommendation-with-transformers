"""Trainer registry."""

from .model_autoencoder_trainer import ModelAutoEncoderTrainer
from .similarity_transformer_trainer import SimilarityTransformerTrainer

TRAINER_REGISTRY = {
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
    "SimilarityTransformerTrainer": SimilarityTransformerTrainer,
}

__all__ = ["TRAINER_REGISTRY", "ModelAutoEncoderTrainer", "SimilarityTransformerTrainer"]
