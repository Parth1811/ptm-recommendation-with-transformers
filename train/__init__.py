"""Trainer registry."""

# from .custom_similarity_transformer_trainer import CustomSimilarityTransformerTrainer
from .model_autoencoder_trainer import ModelAutoEncoderTrainer
from .transformer_trainer import TransformerTrainer

# from .similarity_transformer_trainer import SimilarityTransformerTrainer

TRAINER_REGISTRY = {
#     "CustomSimilarityTransformerTrainer": CustomSimilarityTransformerTrainer,
    "ModelAutoEncoderTrainer": ModelAutoEncoderTrainer,
    "TransformerTrainer": TransformerTrainer,
#     "SimilarityTransformerTrainer": SimilarityTransformerTrainer,
}

__all__ = ["TRAINER_REGISTRY", "ModelAutoEncoderTrainer", "TransformerTrainer"]
