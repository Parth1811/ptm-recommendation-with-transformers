"""Training script for the AutoEncoder over extracted model parameters."""

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


def train_model_autoencoder() -> None:
    """Train the AutoEncoder using parameters stored in .npz archives."""
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

    device = model.device
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = nn.MSELoss()

    logger.info(
        "Starting AutoEncoder training for %d epochs on %d samples.",
        training_config.num_epochs,
        len(dataset),
    )

    for epoch in range(training_config.num_epochs):
        epoch_loss = 0.0
        model.train()

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        logger.info(
            "Epoch %d/%d - reconstruction loss: %.6f",
            epoch + 1,
            training_config.num_epochs,
            epoch_loss,
        )

    save_path: Path = Path(training_config.model_save_directory) / (
        f"autoencoder_weights.loss_{epoch_loss:.6f}.{datetime.now():%Y%m%d_%H%M%S}.pt"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("AutoEncoder weights saved to %s", save_path)
