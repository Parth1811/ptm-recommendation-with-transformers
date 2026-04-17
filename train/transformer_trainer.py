"""Transformer trainer for model-dataset ranking using torch.nn.Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
from beautilog import logger
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from config import ConfigParser, TransformerTrainerConfig
from dataloader import build_combined_similarity_loader
from dataloader.ranking import configure_ranking_paths
from loss import TemperatureScheduler, pairwise_ranking_loss, ranking_loss
from model import CustomSimilarityTransformer, RankingCrossAttentionTransformer

from .base_trainer import BaseTrainer, TrainingMetrics

logger.name = "TransformerTrainer"


class TransformerTrainer(BaseTrainer):
    """Trainer for cross-attention transformer model ranking."""

    def __init__(self) -> None:
        """Initialize TransformerTrainer."""
        # 1. Load config FIRST
        self.config = ConfigParser.get(TransformerTrainerConfig)

        # 1b. Configure ranking data paths if specified
        if self.config.performance_json or self.config.similarity_json:
            configure_ranking_paths(
                performance_json=self.config.performance_json,
                similarity_json=self.config.similarity_json,
            )
            logger.info(
                "Ranking paths: perf=%s, sim=%s",
                self.config.performance_json or "(default)",
                self.config.similarity_json or "(default)",
            )

        # 2. Initialize model based on config
        model_type = getattr(self.config, 'model_type', 'custom_similarity')
        if model_type == 'ranking_cross_attention':
            self.model = RankingCrossAttentionTransformer()
            logger.info("Using RankingCrossAttentionTransformer (6+6 encoder-decoder layers)")
        else:
            self.model = CustomSimilarityTransformer()
            logger.info("Using CustomSimilarityTransformer (stacked cross-attention)")

        # 3. Setup dataloaders with configurable batch size
        self.dataloader = build_combined_similarity_loader(
            split="train", batch_size=self.config.batch_size
        )
        self.val_dataloader = build_combined_similarity_loader(
            split="validation", batch_size=self.config.batch_size
        )

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
        scheduler_type = getattr(self.config, 'scheduler_type', 'reduce_on_plateau')
        if scheduler_type == 'cosine_warm_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=getattr(self.config, 'cosine_t0', 50),
                T_mult=getattr(self.config, 'cosine_t_mult', 2),
                eta_min=self.config.scheduler_min_lr,
            )
            self._scheduler_type = 'cosine_warm_restarts'
            logger.info(f"Using CosineAnnealingWarmRestarts (T_0={getattr(self.config, 'cosine_t0', 50)}, T_mult={getattr(self.config, 'cosine_t_mult', 2)})")
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
                verbose=True,
            )
            self._scheduler_type = 'reduce_on_plateau'

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

    def _forward_batch(self, batch: dict[str, torch.Tensor], is_training: bool = True) -> torch.Tensor:
        """Handle multi-tensor batch with padding masks.

        Args:
            batch: CombinedSimilarityBatch with keys:
                'dataset_tokens': (B, M_max, D) padded
                'dataset_pad_mask': (B, M_max) True where padded
                'model_tokens': (B, N, D)
                'true_ranks': (B, N)
            is_training: If True, advance temperature scheduler. Set False for validation.

        Returns:
            loss: Scalar loss tensor (mean over batch)
        """
        # Move all tensors to device
        dataset_tokens = batch["dataset_tokens"].to(self.device)  # (B, M_max, D)
        model_tokens = batch["model_tokens"].to(self.device)      # (B, N, D)
        true_ranks = batch["true_ranks"].to(self.device)          # (B, N)
        pad_mask = batch.get("dataset_pad_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(self.device)  # (B, M_max)

        # Forward pass - model handles both architectures
        if isinstance(self.model, RankingCrossAttentionTransformer):
            # RankingCrossAttentionTransformer: forward(dataset_tokens, model_tokens)
            # Pass padding mask as src_key_padding_mask
            if pad_mask is not None:
                logits = self.model(
                    dataset_tokens, model_tokens, src_key_padding_mask=pad_mask
                )
            else:
                logits = self.model(dataset_tokens, model_tokens)
        else:
            # CustomSimilarityTransformer: forward(model_tokens, dataset_tokens, dataset_pad_mask)
            logits = self.model(model_tokens, dataset_tokens, dataset_pad_mask=pad_mask)

        # Get current temperature (only advance scheduler during training)
        if self.temp_scheduler is not None and is_training:
            temperature = self.temp_scheduler.step()
        elif self.temp_scheduler is not None:
            temperature = self.temp_scheduler.get_temperature(
                self.temp_scheduler.current_step  # Read without advancing
            )
        else:
            temperature = 1.0

        # Compute ranking loss with temperature scaling
        # ranking_loss returns sum over positions; average over batch
        batch_size = logits.shape[0]
        rank_loss = ranking_loss(logits, true_ranks, reverse_order=True, temperature=temperature)
        rank_loss = rank_loss / batch_size

        if is_training:
            logger.batch(f"Rank Loss: {rank_loss.item():.6f}, Temp: {temperature:.3f}, BS: {batch_size}")
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
                # Scheduler steps differently depending on type
                if self._scheduler_type == 'cosine_warm_restarts':
                    self.scheduler.step(epoch)
                else:
                    self.scheduler.step(val_loss)

                # Check for improvement and early stopping
                # check_early_stopping updates best_val_loss internally
                improved, should_stop = self.check_early_stopping(val_loss)

                if improved:
                    logger.checkpoint(f"New best validation loss: {val_loss:.6f}")
                    self.save_checkpoint(epoch=epoch, is_best=True)

                if should_stop:
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
                loss = self._forward_batch(batch, is_training=False)
                val_loss += loss.item()

        avg_loss = val_loss / len(self.val_dataloader)
        return avg_loss

    def check_early_stopping(self, val_loss: float) -> tuple[bool, bool]:
        """Check if validation loss improved and if early stopping criteria is met.

        Returns:
            (improved, should_stop): Whether loss improved, and whether to stop training.
        """
        if val_loss == -1 or self.config.early_stopping_patience is None:
            return False, False

        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            logger.checkpoint(f"Validation loss improved to {val_loss:.6f}, resetting early stopping counter.")
            self.epochs_without_improvement = 0
            return True, False

        self.epochs_without_improvement += 1
        return False, self.epochs_without_improvement >= self.config.early_stopping_patience

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
