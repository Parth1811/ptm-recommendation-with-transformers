"""Base training utilities."""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from attr import dataclass
from beautilog import logger
from flask import config
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import BaseTrainerConfig
from utils.plot import plot_multiple


@dataclass
class TrainingMetrics:
    """Data class to hold training metrics for a single epoch."""
    epoch: int
    loss: float
    val_loss: float
    other_metrics: Mapping[str, float]


class BaseTrainer(ABC):
    """Base trainer with logging, history tracking, and plotting support."""

    config: BaseTrainerConfig
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: Any
    history: list[TrainingMetrics]


    def __init__(self):
        """Initialize the trainer with config, model, dataloader, and optimizer."""
        if self.config is None or not issubclass(type(self.config), BaseTrainerConfig):
            raise ValueError("Trainer must be initialized with a valid BaseTrainerConfig instance.")

        if self.model is None or not isinstance(self.model, nn.Module):
            raise ValueError("Trainer must be initialized with a valid PyTorch model instance.")

        if self.dataloader is None or not isinstance(self.dataloader, DataLoader):
            raise ValueError("Trainer must be initialized with a valid training DataLoader instance.")

        if self.val_dataloader is None or not isinstance(self.val_dataloader, DataLoader):
            raise ValueError("Trainer must be initialized with a valid validation DataLoader instance.")

        if self.optimizer is None or not isinstance(self.optimizer, torch.optim.Optimizer):
            raise ValueError("Trainer must be initialized with a valid optimizer instance.")

        if self.scheduler is None:
            raise ValueError("Scheduler must be a valid PyTorch learning rate scheduler instance.")

        if self.progress_bar is None or not isinstance(self.progress_bar, tqdm):
            raise ValueError("Must call init_progress_bar() before training.")

        self.history = []
        self.started_at = datetime.now()
        self.run_directory = Path(self.config.model_save_directory) / "runs"
        self.run_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self):
        """Run the training loop for the specified number of epochs."""
        pass

    @abstractmethod
    def save_metrics(self, force_save: bool = False, **kwargs):
        """Should be called to save training metrics to disk after each epoch."""
        pass

    @abstractmethod
    def loss_fn(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute the loss for a given batch of data."""
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save the model checkpoint for a given epoch."""
        pass

    @abstractmethod
    def save_model(self, save_path: str):
        """Save the final model to disk."""
        pass

    def init_progress_bar(self, total: int) -> tqdm:
        """Initialize a tqdm progress bar for training."""
        self.progress_bar = tqdm(
            total=total,
            desc=f"Training {self.__class__.__name__}",
            dynamic_ncols=True,
            leave=False,
            miniters=1,
            colour="yellow",
        )

    def update_progress_bar(self, count, desc: str = None, postfix: dict[str, Any] = None):
        """Update the progress bar with current status."""
        if desc:
            self.progress_bar.set_description(desc)

        if postfix:
            self.progress_bar.set_postfix(postfix)

        self.progress_bar.update(count)

    def get_run_file_path(self, extension: str = "json") -> Path:
        """Get a unique file path for saving training metrics or checkpoints."""
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{self.model.__class__.__name__}_{timestamp}.{extension}"
        return self.run_directory / filename

    def get_model_save_path(self, prefix: str = "", suffix: str = "") -> Path:
        """Get a unique file path for saving the final model."""
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.model.__class__.__name__}_{suffix}_{timestamp}.pt"
        return self.run_directory / filename

    def plot_metrics(self):
        """Plot training and validation metrics over epochs."""
        if not self.history:
            logger.warning("No training history to plot.")
            return

        plots = []
        plots.append({
            'x': [m.epoch for m in self.history],
            'y': [m.loss for m in self.history],
            'title': 'Training Loss over Epochs',
            'xlabel': 'Epoch',
            'ylabel': 'Loss',
            'color': 'blue',
        })

        plots.append({
            'title': 'Log Training Loss over Epochs',
            'x': [m.epoch for m in self.history],
            'y': [m.loss for m in self.history],
            'xlabel': 'Epoch',
            'ylabel': 'Log Loss',
            'yscale': 'log',
            'color': 'orange',
        })

        plots.append({
            'x': [m.epoch for m in self.history],
            'y': [m.val_loss for m in self.history],
            'title': 'Validation Loss over Epochs',
            'xlabel': 'Epoch',
            'ylabel': 'Loss',
        })

        for metric_name in self.history[0].other_metrics.keys():
            plots.append({
                'x': [m.epoch for m in self.history],
                'y': [m.other_metrics[metric_name] for m in self.history],
                'title': f'{metric_name} over epochs',
                'xlabel': 'Epoch',
                'ylabel': metric_name,
            })


        plot_multiple(plots, title='Training Metrics', save_path=self.get_run_file_path('png'))