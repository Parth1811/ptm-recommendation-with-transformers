"""Trainer registry."""

# from .custom_similarity_transformer_trainer import CustomSimilarityTransformerTrainer
from .model_autoencoder_trainer import ModelAutoEncoderTrainer

# from .similarity_transformer_trainer import SimilarityTransformerTrainer

TRAINER_REGISTRY = {
#     "CustomSimilarityTransformerTrainer": CustomSimilarityTransformerTrainer,
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
#     "SimilarityTransformerTrainer": SimilarityTransformerTrainer,
}

__all__ = ["TRAINER_REGISTRY", "ModelAutoEncoderTrainer"]
