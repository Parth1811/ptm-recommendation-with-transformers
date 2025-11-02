"""Training script for the AutoEncoder over extracted model parameters."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from beautilog import logger
from torch import nn
from torch.utils.data import DataLoader

from config import ConfigParser, TrainModelAutoEncoderConfig
from dataloader import ModelParameterDataset
from model import ModelAutoEncoder


class Trainer:
    """Base trainer with logging, history tracking, and plotting support."""

    progress_bar_width = 30

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainModelAutoEncoderConfig,
        *,
        run_name: str | None = None,
        run_directory: Path | None = None,
    ) -> None:
        if getattr(config, "num_epochs", None) is None:
            raise ValueError("Training configuration must provide a num_epochs attribute.")

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = getattr(model, "device", torch.device("cpu"))
        self.num_epochs = config.num_epochs
        self.history: list[dict[str, Any]] = []

        base_run_dir = run_directory or Path(getattr(config, "model_save_directory", Path("artifacts/runs"))) / "runs"
        base_run_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_directory = base_run_dir
        self.run_file_path = self.run_directory / f"{self.run_name}.json"
        self.loss_plot_path = self.run_directory / f"{self.run_name}_loss.png"
        self.started_at = datetime.now().isoformat(timespec="seconds")

        self._serialized_config = self._serialize_config(config)
        logger.info("Initialized trainer %s at %s", self.run_name, self.run_file_path)

    @staticmethod
    def _serialize_config(config: TrainModelAutoEncoderConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in asdict(config).items():
            if isinstance(value, Path):
                payload[key] = str(value)
            else:
                payload[key] = value
        return payload

    def train(self) -> list[dict[str, Any]]:
        """Execute the main training loop."""
        self.on_train_begin()

        for epoch in range(1, self.num_epochs + 1):
            epoch_metrics = self._train_epoch(epoch)
            self.record_data(epoch, epoch_metrics)

        self._save_loss_plot()
        self.on_train_end()
        return self.history

    def on_train_begin(self) -> None:
        """Hook for subclasses to run logic before training starts."""

    def on_train_end(self) -> None:
        """Hook for subclasses to run logic after training ends."""

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Subclasses must implement to return a scalar loss tensor."""
        raise NotImplementedError

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        samples_seen = 0
        total_batches = len(self.dataloader)

        for batch_index, batch in enumerate(self.dataloader, start=1):
            loss_value, batch_size = self._train_batch(batch)
            running_loss += loss_value * batch_size
            samples_seen += batch_size
            self._render_progress(epoch, batch_index, total_batches, loss_value)

        if total_batches:
            print()

        average_loss = running_loss / max(samples_seen, 1)
        return {"loss": float(average_loss)}

    def _train_batch(self, batch: torch.Tensor) -> tuple[float, int]:
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), batch.size(0)

    def _render_progress(self, epoch: int, batch_idx: int, total_batches: int, loss_value: float) -> None:
        progress = batch_idx / max(total_batches, 1)
        filled_length = int(self.progress_bar_width * progress)
        bar = "#" * filled_length + "-" * (self.progress_bar_width - filled_length)
        message = (
            f"\rEpoch {epoch}/{self.num_epochs} "
            f"[{bar}] {progress * 100:5.1f}% "
            f"loss: {loss_value:.4f}"
        )
        print(message, end="", flush=True)

    def record_data(self, epoch: int, metrics: dict[str, float]) -> None:
        """Persist epoch metrics and emit structured logs."""
        record = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.history.append(record)
        logger.info("Epoch %d metrics: %s", epoch, metrics)
        payload = {
            "run_name": self.run_name,
            "created_at": self.started_at,
            "model_class": self.model.__class__.__name__,
            "config": self._serialized_config,
            "history": self.history,
        }
        self.run_file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_loss_plot(self) -> None:
        if not self.history:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping loss plot creation.")
            return

        epochs = [entry["epoch"] for entry in self.history]
        losses = [entry["metrics"]["loss"] for entry in self.history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, losses, marker="o")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.loss_plot_path, dpi=200)
        plt.close()
        logger.info("Saved loss plot to %s", self.loss_plot_path)

    @classmethod
    def replot_metric(
        cls,
        run_file: str | Path,
        metric: str = "loss",
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """Replot a stored metric from a run file."""
        run_path = Path(run_file)
        output = Path(output_path) if output_path else run_path.with_name(f"{run_path.stem}_{metric}.png")

        payload = json.loads(run_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
        if not history:
            raise ValueError(f"No history entries found in {run_path}.")

        x_values = []
        y_values = []
        for entry in history:
            metrics = entry.get("metrics", {})
            if metric not in metrics:
                continue
            x_values.append(entry["epoch"])
            y_values.append(metrics[metric])

        if not x_values:
            raise KeyError(f"Metric '{metric}' not present in run history.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; cannot replot metric %s.", metric)
            return None

        plt.figure(figsize=(8, 4.5))
        plt.plot(x_values, y_values, marker="o")
        plt.title(f"{metric.title()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        logger.info("Saved %s metric plot to %s", metric, output)
        return output


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


def train_model_autoencoder() -> None:
    """Entry point for training the AutoEncoder."""
    trainer = ModelAutoEncoderTrainer()
    trainer.train()
