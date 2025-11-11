"""AutoEncoder training implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from beautilog import logger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ConfigParser, TrainModelAutoEncoderConfig
from dataloader import ModelParameterDataset
from model import ModelAutoEncoder

from .base_trainer import BaseTrainer, TrainingMetrics


class ModelAutoEncoderTrainer(BaseTrainer):
    """Trainer for AutoEncoder models."""

    def __init__(self) -> None:
        """Initialize AutoEncoder trainer."""
        self.config = ConfigParser.get(TrainModelAutoEncoderConfig)
        self.model = ModelAutoEncoder()

        dataset = ModelParameterDataset(root_dir=self.config.extracted_models_dir, normalize=self.config.normalize_inputs,)
        if dataset[0].numel() != self.model.encoder_input_size:
            raise ValueError(f"Input dimension mismatch: dataset sample has {dataset[0].numel()} features but encoder_input_size is {self.model.encoder_input_size}.")

        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle, pin_memory=torch.cuda.is_available(), num_workers=self.config.num_workers)
        self.val_dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=self.config.num_workers)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay,betas=(self.config.beta1, self.config.beta2))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.scheduler_factor, patience=self.config.scheduler_patience, min_lr=self.config.scheduler_min_lr)

        self.init_progress_bar(len(dataset) * self.config.num_epochs)
        super().__init__()

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def loss_fn(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute reconstruction loss with optional L1 penalty on latent codes."""

        if self.config.normalize_inputs:
            batch = (batch - batch.mean()) / (batch.std() + 1e-8)

        code, reconstructed = self.model(batch)

        if self.config.reconstruction_loss == 'smooth_l1':
            reconstruction_loss = nn.SmoothL1Loss(beta=self.config.smooth_l1_beta)(
                reconstructed, batch
            )
        elif self.config.reconstruction_loss == 'mse':
            reconstruction_loss = nn.MSELoss()(reconstructed, batch)
        else:
            reconstruction_loss = nn.L1Loss()(reconstructed, batch)

        l1_penalty = self.config.code_l1_penalty * torch.abs(code).mean()

        return reconstruction_loss + l1_penalty

    def train(self):
        """Run the training loop."""
        self.model.to(self.device)

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch in self.dataloader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                loss = self.loss_fn(batch)
                loss.backward()

                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                self.optimizer.step()
                train_loss += loss.item()
                self.update_progress_bar(len(batch), postfix={'loss': loss.item()})

            train_loss /= len(self.dataloader)

            val_loss = -1
            if epoch % self.config.log_every_n_epochs == 0:
                val_loss = self.validate()

            if epoch % self.config.save_checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch=epoch, is_best=False)

            self.scheduler.step(val_loss)
            self.save_metrics(epoch=epoch, loss=train_loss, val_loss=val_loss,
                                other_metrics={'learning_rate': self.optimizer.param_groups[0]['lr']})

            if self.check_early_stopping(val_loss):
                logger.info(f'Early stopping at epoch {epoch}')
                break

            if val_loss != -1 and val_loss < self.best_val_loss:
                logger.info(f'New best validation loss: {val_loss:.6f} at epoch {epoch}')
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

        self.save_metrics_to_file()
        self.plot_metrics()

    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = batch.to(self.config.device)
                loss = self.loss_fn(batch)
                val_loss += loss.item()

        return val_loss / len(self.val_dataloader)

    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
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
        checkpoint = torch.load(load_path, map_location=self.config.device)
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
            logger.info(
                f"Epoch {kwargs['epoch']}: loss={kwargs['loss']:.6f}, "
                f"val_loss={kwargs['val_loss']:.6f}, "
                f"other_metrics={kwargs.get('other_metrics', {})}"
            )
            self.save_metrics_to_file()