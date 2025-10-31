"""Training script for the AutoEncoder over extracted model parameters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from beautilog import logger
from torch import nn
from torch.utils.data import DataLoader

from config import ConfigParser, ModelAutoEncoderConfig
from dataloader import ModelParameterDataset
from model import AutoEncoder


def train_model_autoencoder() -> None:
    """Train the AutoEncoder using parameters stored in .npz archives."""
    ConfigParser.load()
    autoencoder_config = ConfigParser.get(ModelAutoEncoderConfig)

    dataset = ModelParameterDataset(root_dir=autoencoder_config.extracted_models_dir)
    sample_dimension = dataset[0].numel()
    if sample_dimension != autoencoder_config.encoder_input_size:
        raise ValueError(
            "Input dimension mismatch: dataset sample has "
            f"{sample_dimension} features but encoder_input_size is "
            f"{autoencoder_config.encoder_input_size}."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=autoencoder_config.batch_size,
        shuffle=autoencoder_config.shuffle,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(
        encoder_input_size=autoencoder_config.encoder_input_size,
        encoder_output_size=autoencoder_config.encoder_output_size,
        encoder_hidden_layers=list(autoencoder_config.encoder_hidden_layers),
        decoder_hidden_layers=list(autoencoder_config.decoder_hidden_layers),
        decoder_output_size=autoencoder_config.decoder_output_size,
        use_activation=autoencoder_config.use_activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=autoencoder_config.learning_rate)
    criterion = nn.MSELoss()

    logger.info(
        "Starting AutoEncoder training for %d epochs on %d samples.",
        autoencoder_config.num_epochs,
        len(dataset),
    )

    for epoch in range(autoencoder_config.num_epochs):
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
        logger.info("Epoch %d/%d - reconstruction loss: %.6f", epoch + 1, autoencoder_config.num_epochs, epoch_loss)

    save_path: Path = Path(autoencoder_config.model_save_directory) / f"autoencoder_weights.loss_{epoch_loss:.6f}.{datetime.now():%Y%m%d_%H%M%S}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("AutoEncoder weights saved to %s", save_path)
