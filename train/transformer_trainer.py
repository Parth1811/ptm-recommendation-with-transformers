"""Transformer trainer for model-dataset ranking using torch.nn.Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
from beautilog import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ConfigParser, TransformerTrainerConfig
from dataloader import build_combined_similarity_loader
from loss import TemperatureScheduler, ranking_loss
from model import RankingCrossAttentionTransformer

from .base_trainer import BaseTrainer, TrainingMetrics

logger.name = "TransformerTrainer"


class TransformerTrainer(BaseTrainer):
    """Trainer for cross-attention transformer model ranking."""

    def __init__(self) -> None:
        """Initialize TransformerTrainer."""
        # 1. Load config FIRST
        self.config = ConfigParser.get(TransformerTrainerConfig)

        # 2. Initialize model
        self.model = RankingCrossAttentionTransformer()

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

        # 6. Initialize temperature scheduler
        total_steps = len(self.dataloader) * self.config.num_epochs
        if self.config.use_temperature_scheduler:
            self.temp_scheduler = TemperatureScheduler(
                initial_temp=self.config.initial_temperature,
                final_temp=self.config.final_temperature,
                total_steps=total_steps,
                schedule=self.config.temperature_schedule,
                warmup_steps=self.config.temperature_warmup_steps,
            )
            logger.info(
                f"Temperature scheduler initialized: {self.config.initial_temperature} -> "
                f"{self.config.final_temperature} over {total_steps} steps ({self.config.temperature_schedule})"
            )
        else:
            self.temp_scheduler = None

        # 7. Initialize progress bar
        self.init_progress_bar(total=total_steps)

        # 8. Call super().__init__() LAST
        super().__init__()

        # 9. Load from checkpoint if configured
        if self.config.load_from_checkpoint and self.config.checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {self.config.checkpoint_path}")
            self.load_checkpoint(
                load_path=self.config.checkpoint_path,
                only_model_weights=self.config.only_load_model_weights
            )

        # 10. Additional state variables
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

        # Get current temperature from scheduler
        if self.temp_scheduler is not None:
            temperature = self.temp_scheduler.step()
        else:
            temperature = 1.0

        # Compute combined loss
        # 1. Ranking loss (listwise) with temperature scaling
        rank_loss = ranking_loss(logits, true_ranks, reverse_order=True, temperature=temperature)

        # 2. Smooth L1 loss for stability (optional regularization)
        # Convert ranks to target scores (inverse of rank for regression)
        # target_scores = self.config.num_models - true_ranks.float()
        # smooth_l1_loss = nn.SmoothL1Loss()(logits, target_scores)

        # Combined loss
        # total_loss = (
        #     self.config.ranking_loss_weight * rank_loss +
        #     self.config.smooth_l1_weight * smooth_l1_loss
        # )

        # Note: L2 regularization is handled by optimizer weight_decay parameter
        # No need for manual regularization loss

        logger.batch(f"Rank Loss: {rank_loss.item():.6f}, Temp: {temperature:.3f}")
        return rank_loss

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


            # Save periodic checkpoints
            if epoch % self.config.save_checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch=epoch, is_best=False)

            # Save metrics
            other_metrics = {'learning_rate': self.optimizer.param_groups[0]['lr']}
            if self.temp_scheduler is not None:
                other_metrics['temperature'] = self.temp_scheduler.get_temperature(
                    self.temp_scheduler.current_step - 1  # Get last used temperature
                )


            # Run Validation on every n epochs
            val_loss = -1.0
            if epoch % self.config.validate_every_n_epochs == 0:
                val_loss = self.validate()

            # Save metrics for every epoch, including validation loss if available
            self.save_metrics(
                epoch=epoch,
                loss=avg_train_loss,
                val_loss=val_loss,
                other_metrics=other_metrics
            )

            # Scheduler steps on validation loss every n epochs
            if epoch % self.config.validate_every_n_epochs == 0:
                # Scheduler steps on VALIDATION loss
                self.scheduler.step(val_loss)

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

        # Save temperature scheduler state if enabled
        if self.temp_scheduler is not None:
            checkpoint['temp_scheduler_state_dict'] = self.temp_scheduler.state_dict()

        if is_best:
            torch.save(checkpoint, self.get_model_save_path(suffix="best"))

        torch.save(checkpoint, self.get_model_save_path(prefix="checkpoint", suffix=f"epoch_{epoch}"))

    def save_model(self, save_path):
        """Save the final model to disk."""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, load_path, only_model_weights: bool = False):
        """Load model weights from disk."""
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not only_model_weights:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state tensors to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Move scheduler state tensors to the correct device
            if hasattr(self.scheduler, 'state_dict'):
                for key, value in self.scheduler.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.scheduler, key, value.to(self.device))

            # Load temperature scheduler state if available
            if self.temp_scheduler is not None and 'temp_scheduler_state_dict' in checkpoint:
                self.temp_scheduler.load_state_dict(checkpoint['temp_scheduler_state_dict'])
                # Move temperature scheduler state tensors to the correct device
                if hasattr(self.temp_scheduler, '__dict__'):
                    for key, value in self.temp_scheduler.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            setattr(self.temp_scheduler, key, value.to(self.device))

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
            self.plot_metrics()
