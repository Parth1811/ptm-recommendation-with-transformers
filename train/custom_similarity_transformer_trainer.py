"""Trainer for the CustomSimilarityTransformer model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from beautilog import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ConfigParser, CustomSimilarityTransformerConfig, CustomSimilarityTransformerTrainerConfig
from dataloader import build_dataset_token_loader
from loss.ranking_loss import ranking_loss
from model import CustomSimilarityTransformer

from .base_trainer import Trainer


class CustomSimilarityTransformerTrainer(Trainer):
    """Trainer for CustomSimilarityTransformer model using ranking and regularization losses.

    This trainer:
    - Extracts model_tokens and dataset_tokens from dataloader batches
    - Passes them through CustomSimilarityTransformer to get probability distributions
    - Computes combined loss: ranking_loss + regularization_weight * l2_loss
    - Automatically saves model weights at training end
    """

    def __init__(self) -> None:
        """Initialize the trainer with config, model, dataloader, and optimizer."""
        training_config = ConfigParser.get(CustomSimilarityTransformerTrainerConfig)
        model_config = ConfigParser.get(CustomSimilarityTransformerConfig)

        # Initialize model
        model = CustomSimilarityTransformer(config=model_config)

        # Build dataloaders
        train_dataloader = build_dataset_token_loader(
            splits=("train",),
            shuffle=training_config.shuffle,
        )
        val_dataloader = build_dataset_token_loader(
            splits=("validation",),
            shuffle=False,
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Initialize scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=training_config.scheduler_mode,
            factor=training_config.scheduler_factor,
            patience=training_config.scheduler_patience,
            min_lr=training_config.scheduler_min_lr,
            cooldown=training_config.scheduler_cooldown,
            threshold=training_config.scheduler_threshold,
            threshold_mode=training_config.scheduler_threshold_mode,
        )

        # Call parent constructor
        super().__init__(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            config=training_config,
            scheduler=scheduler,
            val_dataloader=val_dataloader,
        )

        # Move model to device
        self.model.to(self.device)

        # Store configuration and loss weight
        self.regularization_weight = float(training_config.regularization_weight)
        self._last_batch_metrics: dict[str, float] = {}

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute combined ranking and regularization loss.

        Args:
            batch: Dictionary containing:
                - dataset_tokens: Tensor of shape (B, M, D) - dataset feature tokens
                - model_tokens: Tensor of shape (B, N, D) - model embeddings
                - true_ranks: Tensor of shape (B, N) - true ranking scores/indices
                Additional keys may be present (metadata).

        Returns:
            Scalar loss tensor combining ranking and regularization losses.

        Raises:
            KeyError: If required batch keys are missing.
            ValueError: If tensor shapes are invalid.
        """
        # Extract required tensors from batch
        if "dataset_tokens" not in batch:
            raise KeyError("Batch must contain 'dataset_tokens'")
        if "model_tokens" not in batch:
            raise KeyError("Batch must contain 'model_tokens'")
        if "true_ranks" not in batch:
            raise KeyError("Batch must contain 'true_ranks'")

        dataset_tokens = batch["dataset_tokens"]
        model_tokens = batch["model_tokens"]
        true_ranks = batch["true_ranks"]

        # Validate shapes
        if dataset_tokens.dim() != 3:
            raise ValueError(
                f"Expected dataset_tokens to have 3 dimensions [B, M, D], "
                f"got shape {dataset_tokens.shape}"
            )
        if model_tokens.dim() != 3:
            raise ValueError(
                f"Expected model_tokens to have 3 dimensions [B, N, D], "
                f"got shape {model_tokens.shape}"
            )

        # Forward pass through model to get probability distribution
        # Output shape: (B, N) - probability distribution over N models
        probs = self.model(model_tokens, dataset_tokens)

        # Compute ranking loss comparing predicted probs against true ranks
        batch_size = probs.size(0)

        ranking_losses = []
        for b in range(batch_size):
            pred_scores = probs[b]  # Shape: (N,)
            true_rank = true_ranks[b]  # Shape: (N,)

            # ranking_loss expects 1D tensors
            loss = ranking_loss(pred_scores, true_rank)
            ranking_losses.append(loss)

        if ranking_losses:
            ranking_loss_value = torch.stack(ranking_losses).mean()
        else:
            ranking_loss_value = torch.tensor(0.0, device=self.device)

        # Compute L2 regularization on model parameters
        l2_loss = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_loss = l2_loss + torch.norm(param, p=2)

        # Combine losses
        total_loss = ranking_loss_value + self.regularization_weight * l2_loss

        # Store metrics for logging
        self._last_batch_metrics = {
            "ranking_loss": float(ranking_loss_value.detach().item()),
            "l2_loss": float(l2_loss.detach().item()),
        }

        return total_loss

    def _collect_batch_metrics(self) -> dict[str, float]:
        """Return metrics collected during batch processing.

        Returns:
            Dictionary of metric name to scalar value.
        """
        return dict(self._last_batch_metrics)

    def on_train_begin(self) -> None:
        """Log training configuration and model setup at start of training."""
        logger.info(
            "Starting CustomSimilarityTransformer training for %d epochs | "
            "lr=%.2e | weight_decay=%.1e | regularization_weight=%.3f | grad_clip=%s",
            self.config.num_epochs,
            self.config.learning_rate,
            self.config.weight_decay,
            self.regularization_weight,
            f"{self.gradient_clip_norm:.2f}" if self.gradient_clip_norm is not None else "disabled",
        )
        logger.info(
            "Model: embed_dim=%d num_heads=%d num_layers=%d | "
            "Device: %s | Early stopping patience: %s",
            self.model.embed_dim,
            self.model.num_heads,
            self.model.num_layers,
            self.device,
            self.early_stopping_patience or "disabled",
        )

    def on_train_end(self) -> None:
        """Save model weights with timestamp and final loss in filename."""
        if not self.history:
            logger.warning("No training history; skipping model save.")
            return

        # Get final loss from history
        final_metrics = self.history[-1]["metrics"]
        final_loss = final_metrics.get("loss")
        if final_loss is None:
            logger.warning("Final loss missing; skipping model save.")
            return

        # Create save directory
        save_dir = Path(self.config.model_save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model with timestamp and loss in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"custom_similarity_transformer_weights.loss_{final_loss:.6f}.{timestamp}.pt"

        torch.save(self.model.state_dict(), save_path)
        logger.info("Model weights saved to %s", save_path)

        # Log final metrics
        logger.info("Training completed. Final metrics: %s", final_metrics)
