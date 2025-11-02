"""Model AutoEncoder trainer definition."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from beautilog import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ConfigParser, TrainModelAutoEncoderConfig
from dataloader import ModelParameterDataset
from model import ModelAutoEncoder

from .base_trainer import Trainer


class ModelAutoEncoderTrainer(Trainer):
    """Trainer specialization for the AutoEncoder model."""

    def __init__(self) -> None:
        ConfigParser.load()
        training_config = ConfigParser.get(TrainModelAutoEncoderConfig)
        model = ModelAutoEncoder()

        dataset = ModelParameterDataset(
            root_dir=training_config.extracted_models_dir,
            normalize=training_config.normalize_inputs,
        )
        sample_dimension = dataset[0].numel()
        if sample_dimension != model.encoder_input_size:
            raise ValueError(f"Input dimension mismatch: dataset sample has {sample_dimension} features but encoder_input_size is {model.encoder_input_size}.")

        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=training_config.shuffle,
            pin_memory=torch.cuda.is_available(),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=training_config.weight_decay,
        )
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

        super().__init__(model=model, dataloader=dataloader, optimizer=optimizer, config=training_config, scheduler=scheduler)
        self.criterion = self._build_loss(training_config)
        self.dataset_size = len(dataset)
        self.code_l1_penalty = float(training_config.code_l1_penalty)
        self._last_encoded_stats: dict[str, float] = {}

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        encoded, reconstructed = self.model(batch)
        loss = self.criterion(reconstructed, batch)
        if self.code_l1_penalty > 0.0:
            loss = loss + self.code_l1_penalty * encoded.abs().mean()

        encoded_detached = encoded.detach()
        code_mean = float(encoded_detached.mean().item())
        code_abs_mean = float(encoded_detached.abs().mean().item())
        if encoded_detached.numel() > 1:
            code_std = float(encoded_detached.std(unbiased=False).item())
        else:
            code_std = 0.0

        self._last_encoded_stats = {
            "code_mean": code_mean,
            "code_abs_mean": code_abs_mean,
            "code_std": code_std,
        }

        return loss

    def _collect_batch_metrics(self) -> dict[str, float]:
        return self._last_encoded_stats

    def on_train_begin(self) -> None:
        logger.info(
            "Starting AutoEncoder training for %d epochs on %d samples (batch_size=%d, lr=%.3g, weight_decay=%.1e).",
            self.config.num_epochs,
            self.dataset_size,
            self.config.batch_size,
            self.config.learning_rate,
            self.config.weight_decay,
        )
        clip_display = (
            f"{self.config.gradient_clip_norm:.2f}" if self.config.gradient_clip_norm is not None else "disabled"
        )

        logger.info(
            "Normalization=%s | Reconstruction loss=%s | SmoothL1 beta=%.2f | Code L1 penalty=%.1e | Grad clip=%s",
            "enabled" if self.config.normalize_inputs else "disabled",
            self.config.reconstruction_loss,
            self.config.smooth_l1_beta,
            self.code_l1_penalty,
            clip_display,
        )

    def on_train_end(self) -> None:
        if not self.history:
            return

        final_loss = self.history[-1]["metrics"]["loss"]
        save_path: Path = Path(self.config.model_save_directory) / (f"autoencoder_weights.loss_{final_loss:.6f}.{datetime.now():%Y%m%d_%H%M%S}.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info("AutoEncoder weights saved to %s", save_path)

    @staticmethod
    def _build_loss(config: TrainModelAutoEncoderConfig) -> nn.Module:
        loss_type = getattr(config, "reconstruction_loss", "smooth_l1").lower()
        if loss_type == "smooth_l1":
            return nn.SmoothL1Loss(beta=config.smooth_l1_beta, reduction="mean")
        if loss_type == "mae":
            return nn.L1Loss(reduction="mean")
        if loss_type == "mse":
            return nn.MSELoss(reduction="mean")
        raise ValueError(f"Unsupported reconstruction loss: {config.reconstruction_loss}")
