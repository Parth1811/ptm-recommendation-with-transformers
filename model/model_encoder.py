"""Configurable autoencoder implementation."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from config import ConfigParser, ModelAutoEncoderConfig


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
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        decoder_output_size = decoder_output_size or encoder_input_size

        self.encoder_input_size = encoder_input_size
        self.encoder_output_size = encoder_output_size
        self.decoder_output_size = decoder_output_size
        self.use_activation = use_activation
        self.dropout = max(0.0, float(dropout))
        self.activation_name = activation.lower()
        self.activation_factory = self._resolve_activation(self.activation_name)

        self.encoder = self._build_mlp(
            input_dim=encoder_input_size,
            hidden_layers=encoder_hidden_layers or (),
            output_dim=encoder_output_size,
            use_activation=use_activation,
            dropout=self.dropout,
            activation_factory=self.activation_factory,
        )
        self.decoder = self._build_mlp(
            input_dim=encoder_output_size,
            hidden_layers=decoder_hidden_layers or (),
            output_dim=decoder_output_size,
            use_activation=use_activation,
            dropout=self.dropout,
            activation_factory=self.activation_factory,
        )

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int,
        use_activation: bool,
        dropout: float,
        activation_factory: Callable[[], nn.Module],
    ) -> nn.Sequential:
        """Construct a feed-forward stack of Linear layers."""
        layers: list[nn.Module] = []
        previous_dim = input_dim
        all_layers = list(hidden_layers) + [output_dim]

        for index, layer_dim in enumerate(all_layers):
            layers.append(nn.Linear(previous_dim, layer_dim))
            is_last_layer = index == len(all_layers) - 1
            if not is_last_layer:
                if use_activation:
                    layers.append(activation_factory())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            previous_dim = layer_dim

        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_activation(name: str) -> Callable[[], nn.Module]:
        lowered = name.lower()
        if lowered == "relu":
            return nn.ReLU
        if lowered == "gelu":
            return nn.GELU
        if lowered in {"leaky_relu", "leakyrelu"}:
            return lambda: nn.LeakyReLU(negative_slope=0.01)
        raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs and reconstruct them."""
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed


class ModelAutoEncoder(AutoEncoder):
    """AutoEncoder initialized directly from configuration values."""

    def __init__(self, config: ModelAutoEncoderConfig | None = None, *, device: torch.device | str | None = None, auto_configure_device: bool = True) -> None:
        self.config = config or ConfigParser.get(ModelAutoEncoderConfig)
        resolved_device = self._resolve_device(device) if auto_configure_device else device

        super().__init__(
            encoder_input_size=self.config.encoder_input_size,
            encoder_output_size=self.config.encoder_output_size,
            encoder_hidden_layers=list(self.config.encoder_hidden_layers),
            decoder_hidden_layers=list(self.config.decoder_hidden_layers),
            decoder_output_size=self.config.decoder_output_size,
            use_activation=self.config.use_activation,
            dropout=self.config.dropout,
            activation=self.config.activation,
        )

        if resolved_device is not None:
            self.device = torch.device(resolved_device)
            self.to(self.device)
        else:
            self.device = torch.device("cpu")

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)
