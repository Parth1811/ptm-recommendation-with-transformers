"""Model AutoEncoder trainer definition."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from beautilog import logger
from torch import nn
from torch.utils.data import DataLoader

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

        dataset = ModelParameterDataset(root_dir=training_config.extracted_models_dir)
        sample_dimension = dataset[0].numel()
        if sample_dimension != model.encoder_input_size:
            raise ValueError(
                "Input dimension mismatch: dataset sample has "
                f"{sample_dimension} features but encoder_input_size is "
                f"{model.encoder_input_size}."
            )

        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=training_config.shuffle,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            config=training_config,
        )
        self.criterion = nn.MSELoss()
        self.dataset_size = len(dataset)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        _, reconstructed = self.model(batch)
        return self.criterion(reconstructed, batch)

    def on_train_begin(self) -> None:
        logger.info(
            "Starting AutoEncoder training for %d epochs on %d samples.",
            self.config.num_epochs,
            self.dataset_size,
        )

    def on_train_end(self) -> None:
        if not self.history:
            return

        final_loss = self.history[-1]["metrics"]["loss"]
        save_path: Path = Path(self.config.model_save_directory) / (
            f"autoencoder_weights.loss_{final_loss:.6f}.{datetime.now():%Y%m%d_%H%M%S}.pt"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info("AutoEncoder weights saved to %s", save_path)

