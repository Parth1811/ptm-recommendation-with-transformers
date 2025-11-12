"""Transformer trainer for model-dataset ranking using torch.nn.Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
from beautilog import logger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ConfigParser, TransformerTrainerConfig
from dataloader import build_combined_similarity_loader
from loss.ranking_loss import ranking_loss

from .base_trainer import BaseTrainer, TrainingMetrics

logger.name = "TransformerTrainer"


class CrossAttentionTransformer(nn.Module):
    """Cross-attention transformer for model-dataset ranking.

    Uses dataset_tokens as source (encoder input) and model_tokens as target (decoder input).
    Outputs logits for ranking the models.
    """

    def __init__(self, config: TransformerTrainerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_models = config.num_models

        # Transformer with cross-attention
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projection to scalar scores per model
        self.output_projection = nn.Linear(config.d_model, 1)

    def forward(
        self,
        dataset_tokens: torch.Tensor,
        model_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through cross-attention transformer.

        Args:
            dataset_tokens: Shape (batch_size, seq_len, d_model) - dataset features
            model_tokens: Shape (batch_size, num_models, d_model) - model embeddings

        Returns:
            logits: Shape (batch_size, num_models) - ranking scores
        """
        # Transformer expects (batch_size, seq_len, d_model) for both inputs
        # dataset_tokens -> encoder (source/memory)
        # model_tokens -> decoder (target)

        # Pass through transformer with cross-attention
        # output shape: (batch_size, num_models, d_model)
        output = self.transformer(
            src=dataset_tokens,  # Encoder input
            tgt=model_tokens,    # Decoder input (attends to encoder via cross-attention)
        )

        # Project to scalar scores: (batch_size, num_models, d_model) -> (batch_size, num_models, 1)
        logits = self.output_projection(output)

        # Squeeze last dimension: (batch_size, num_models, 1) -> (batch_size, num_models)
        logits = logits.squeeze(-1)
        logits = torch.softmax(logits, dim=-1)  # Optional: convert to probabilities

        return logits


class TransformerTrainer(BaseTrainer):
    """Trainer for cross-attention transformer model ranking."""

    def __init__(self) -> None:
        """Initialize TransformerTrainer."""
        # 1. Load config FIRST
        self.config = ConfigParser.get(TransformerTrainerConfig)

        # 2. Initialize model
        self.model = CrossAttentionTransformer(self.config)

        # 3. Setup dataloaders
        self.dataloader = build_combined_similarity_loader(split="train")
        self.val_dataloader = build_combined_similarity_loader(split="validation")

        # Validate batch structure
        sample_batch = next(iter(self.dataloader))
        logger.info(
            f"Batch structure - dataset_tokens: {sample_batch['dataset_tokens'].shape}, "
            f"model_tokens: {sample_batch['model_tokens'].shape}, "
            f"true_ranks: {sample_batch['true_ranks'].shape}"
        )

        # 4. Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 5. Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
            verbose=True,
        )

        # 6. Initialize progress bar
        total_steps = len(self.dataloader) * self.config.num_epochs
        self.init_progress_bar(total=total_steps)

        # 7. Call super().__init__() LAST
        super().__init__()

        # 8. Additional state variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Handle multi-tensor batch with custom preprocessing.

        Args:
            batch: Dict with keys 'dataset_tokens', 'model_tokens', 'true_ranks'

        Returns:
            loss: Scalar loss tensor
        """
        # Move all tensors to device
        dataset_tokens = batch["dataset_tokens"].to(self.device)  # (B, seq_len, d_model)
        model_tokens = batch["model_tokens"].to(self.device)      # (B, num_models, d_model)
        true_ranks = batch["true_ranks"].to(self.device)          # (B, num_models)

        # Handle batch dimension properly
        # The dataloader returns batch_size=1, so squeeze if needed
        if dataset_tokens.dim() == 4:  # (1, batches, num_classes, dim)
            dataset_tokens = dataset_tokens.squeeze(0)  # (batches, num_classes, dim)

        # Forward pass through transformer
        # logits shape: (batch_size, num_models)
        logits = self.model(dataset_tokens, model_tokens)

        # Compute combined loss
        # 1. Ranking loss (listwise)
        rank_loss = ranking_loss(logits, true_ranks, reverse_order=True)

        # 2. Smooth L1 loss for stability (optional regularization)
        # Convert ranks to target scores (inverse of rank for regression)
        # target_scores = self.config.num_models - true_ranks.float()
        # smooth_l1_loss = nn.SmoothL1Loss()(logits, target_scores)

        # Combined loss
        # total_loss = (
        #     self.config.ranking_loss_weight * rank_loss +
        #     self.config.smooth_l1_weight * smooth_l1_loss
        # )

        # 3. Add regularization loss if needed (e.g., L2 on model parameters)
        reg_loss = 0.0
        if self.config.weight_decay > 0:
            reg_loss = sum(param.norm(2) ** 2 for param in self.model.parameters())
            reg_loss = self.config.weight_decay * reg_loss

        logger.batch(f"Rank Loss: {rank_loss.item():.6f}, Reg Loss: {reg_loss.item():.6f}")
        return (rank_loss + reg_loss)

    def loss_fn(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss (legacy method for compatibility)."""
        return self._forward_batch(batch)

    def train(self):
        """Run the training loop."""
        self.model.to(self.device)
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                # BaseTrainer handles device placement via hooks
                loss = self._forward_batch(batch)
                logger.batch(f"Epoch {epoch} - Batch Loss: {loss.item():.6f}")
                loss.backward()

                # Gradient clipping if configured
                if self.config.gradient_clip_norm and self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                self.optimizer.step()
                train_loss += loss.item()

                self.update_progress_bar(1, postfix={'loss': loss.item()})

            avg_train_loss = train_loss / len(self.dataloader)

            # Validate every N epochs
            val_loss = -1
            if epoch % self.config.validate_every_n_epochs == 0:
                val_loss = self.validate()

                # Scheduler steps on VALIDATION loss
                self.scheduler.step(val_loss)

                # Save if best
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    logger.checkpoint(f'New best validation loss: {val_loss:.6f} at epoch {epoch}')

            # Save periodic checkpoints
            if epoch % self.config.save_checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch=epoch, is_best=False)

            # Save metrics
            self.save_metrics(
                epoch=epoch,
                loss=avg_train_loss,
                val_loss=val_loss,
                other_metrics={'learning_rate': self.optimizer.param_groups[0]['lr']}
            )

            # Check early stopping
            if self.check_early_stopping(val_loss):
                logger.epoch(f'Early stopping at epoch {epoch}')
                break

        self.save_metrics_to_file()
        self.plot_metrics()

    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._forward_batch(batch)
                val_loss += loss.item()

        avg_loss = val_loss / len(self.val_dataloader)
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        if val_loss == -1 or self.config.early_stopping_patience is None:
            return False

        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.config.early_stopping_patience

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config.model_save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if is_best:
            torch.save(checkpoint, self.get_model_save_path(suffix="best"))

        torch.save(checkpoint, self.get_model_save_path(prefix="checkpoint", suffix=f"epoch_{epoch}"))

    def save_model(self, save_path):
        """Save the final model to disk."""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, load_path):
        """Load model weights from disk."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    def save_metrics(self, force_save: bool = False, **kwargs):
        """Save training metrics to disk."""
        self.history.append(TrainingMetrics(
            epoch=kwargs.get('epoch'),
            loss=kwargs.get('loss'),
            val_loss=kwargs.get('val_loss'),
            other_metrics=kwargs.get('other_metrics', {}),
        ))

        if force_save or (kwargs.get('epoch') is not None and kwargs.get('epoch') % self.config.log_every_n_epochs == 0):
            logger.epoch(
                f"Epoch {kwargs['epoch']}: loss={kwargs['loss']:.6f}, "
                f"val_loss={kwargs['val_loss']:.6f}, "
                f"other_metrics={kwargs.get('other_metrics', {})}"
            )
            self.save_metrics_to_file()
