"""Configurable autoencoder implementation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """Simple configurable autoencoder built from linear layers."""

    def __init__(
        self,
        encoder_input_size: int,
        encoder_output_size: int,
        encoder_hidden_layers: Sequence[int] | None = None,
        decoder_hidden_layers: Sequence[int] | None = None,
        decoder_output_size: int | None = None,
        *,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        decoder_output_size = decoder_output_size or encoder_input_size

        self.encoder_input_size = encoder_input_size
        self.encoder_output_size = encoder_output_size
        self.decoder_output_size = decoder_output_size
        self.use_activation = use_activation

        self.encoder = self._build_mlp(
            input_dim=encoder_input_size,
            hidden_layers=encoder_hidden_layers or (),
            output_dim=encoder_output_size,
            use_activation=use_activation,
        )
        self.decoder = self._build_mlp(
            input_dim=encoder_output_size,
            hidden_layers=decoder_hidden_layers or (),
            output_dim=decoder_output_size,
            use_activation=use_activation,
        )

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int,
        use_activation: bool,
    ) -> nn.Sequential:
        """Construct a feed-forward stack of Linear layers."""
        layers: list[nn.Module] = []
        previous_dim = input_dim
        all_layers = list(hidden_layers) + [output_dim]

        for index, layer_dim in enumerate(all_layers):
            layers.append(nn.Linear(previous_dim, layer_dim))
            is_last_layer = index == len(all_layers) - 1
            if use_activation and not is_last_layer:
                layers.append(nn.ReLU())
            previous_dim = layer_dim

        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs and reconstruct them."""
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed
