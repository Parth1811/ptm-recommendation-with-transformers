"""Base training utilities."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from beautilog import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import BaseTrainerConfig
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class Trainer:
    """Base trainer with logging, history tracking, and plotting support."""

    progress_bar_colour: str = "yellow"

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader | Iterable[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        config: BaseTrainerConfig,
        *,
        run_name: str | None = None,
        run_directory: Path | None = None,
        scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    ) -> None:
        if getattr(config, "num_epochs", None) is None:
            raise ValueError("Training configuration must provide a num_epochs attribute.")

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = getattr(model, "device", torch.device("cpu"))
        self.num_epochs = config.num_epochs
        self.history: list[dict[str, Any]] = []
        self.gradient_clip_norm = getattr(config, "gradient_clip_norm", None)
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)

        self.early_stopping_patience = getattr(config, "early_stopping_patience", None)
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            self.early_stopping_patience = None
        self.early_stopping_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
        self._best_monitored_loss = float("inf")
        self._epochs_without_improvement = 0
        self._stop_requested = False

        base_run_dir = run_directory or Path(getattr(config, "model_save_directory", Path("artifacts/runs"))) / "runs"
        base_run_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_directory = base_run_dir
        self.run_file_path = self.run_directory / f"{self.run_name}.json"
        self.metrics_plot_path = self.run_directory / f"{self.run_name}_metrics.png"
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self._elapsed_seconds: float | None = None
        self._timer_start: float | None = None

        self._log_every_n_epochs = max(1, getattr(config, "log_every_n_epochs", 1))
        logger.info("Initialized trainer %s at %s", self.run_name, self.run_file_path)


    def _persist_run_file(self) -> None:
        payload: dict[str, Any] = {
            "run_name": self.run_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_class": self.model.__class__.__name__,
            "history": self.history,
            "metrics_plot_file": self.metrics_plot_path.name,
        }

        if self._elapsed_seconds is not None:
            payload["total_training_seconds"] = self._elapsed_seconds

        self.run_file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def train(self) -> list[dict[str, Any]]:
        """Execute the main training loop."""
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._timer_start = time.perf_counter()
        self._persist_run_file()
        self.on_train_begin()

        total_batches = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        total_steps = (
            max(self.num_epochs * max(total_batches, 1), 1) if total_batches is not None else None
        )
        progress_desc = getattr(self.config, "progress_description", self.model.__class__.__name__)

        progress_bar = tqdm(
            total=total_steps,
            desc=f"Training {progress_desc}",
            ascii=False,
            dynamic_ncols=True,
            colour=self.progress_bar_colour,
        )

        for epoch in range(1, self.num_epochs + 1):
            progress_bar.set_description(f"Training {progress_desc} (Epoch {epoch}/{self.num_epochs})")
            epoch_metrics = self._train_epoch(epoch, progress_bar, total_batches)
            self.record_data(epoch, epoch_metrics)
            stop_training = self._after_epoch(epoch, epoch_metrics)
            if self._should_log_epoch(epoch):
                logger.info("Epoch %d metrics: %s", epoch, epoch_metrics)
            if stop_training:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        progress_bar.close()
        self._save_metric_plots()
        self.completed_at = datetime.now().isoformat(timespec="seconds")
        if self._timer_start is not None:
            self._elapsed_seconds = time.perf_counter() - self._timer_start
        self._persist_run_file()
        self.on_train_end()
        return self.history

    def on_train_begin(self) -> None:
        """Hook for subclasses to run logic before training starts."""

    def on_train_end(self) -> None:
        """Hook for subclasses to run logic after training ends."""

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Subclasses must implement to return a scalar loss tensor."""
        raise NotImplementedError

    def _train_epoch(self, epoch: int, progress_bar: tqdm, total_batches: int | None) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        samples_seen = 0
        grad_norms: list[float] = []
        aggregated_metrics: dict[str, float] = {}
        aggregated_counts: dict[str, int] = {}

        dataloader_iterable: Iterable[torch.Tensor] = self.dataloader

        for batch_index, batch in enumerate(dataloader_iterable, start=1):
            loss_value, batch_size, grad_norm, batch_metrics = self._train_batch(batch)
            running_loss += loss_value * batch_size
            samples_seen += batch_size
            if grad_norm is not None:
                grad_norms.append(grad_norm)
            for key, value in batch_metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
                aggregated_counts[key] = aggregated_counts.get(key, 0) + 1

            progress_bar.update(1)
            progress_bar.set_postfix({"epoch": epoch, "loss": f"{loss_value:.4f}"}, refresh=False)

        if total_batches is not None and total_batches == 0:
            logger.warning("No batches available for training; did you provide an empty dataset?")

        average_loss = running_loss / max(samples_seen, 1)
        metrics: dict[str, float] = {"loss": float(average_loss)}
        if grad_norms:
            metrics["grad_norm"] = float(np.mean(grad_norms))
        for key, total in aggregated_metrics.items():
            metrics[key] = float(total / max(aggregated_counts.get(key, 1), 1))
        return metrics

    def _train_batch(self, batch: torch.Tensor) -> tuple[float, int, float | None, dict[str, float]]:
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(batch)
        loss.backward()
        grad_norm = self._apply_gradient_clipping()
        self.optimizer.step()
        batch_metrics = self._collect_batch_metrics()
        return loss.detach().item(), batch.size(0), grad_norm, batch_metrics

    def _apply_gradient_clipping(self) -> float | None:
        if self.gradient_clip_norm is None:
            return self._compute_total_grad_norm()
        result = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        if isinstance(result, torch.Tensor):
            return float(result.item())
        return float(result)

    def _compute_total_grad_norm(self) -> float | None:
        total_sq = 0.0
        has_grad = False
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            has_grad = True
            grad = parameter.grad.detach()
            total_sq += float(grad.pow(2).sum().item())
        if not has_grad:
            return None
        return float(total_sq ** 0.5)

    def _collect_batch_metrics(self) -> dict[str, float]:
        """Hook for subclasses to expose batch level metrics."""
        return {}

    def record_data(self, epoch: int, metrics: dict[str, float]) -> None:
        """Persist epoch metrics and emit structured logs."""
        record = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.history.append(record)
        self._persist_run_file()

    def _save_metric_plots(self) -> None:
        if not self.history:
            return

        epochs = np.array([entry["epoch"] for entry in self.history], dtype=float)
        losses = np.array([entry["metrics"]["loss"] for entry in self.history], dtype=float)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        axes = np.atleast_1d(axes)

        min_loss_idx = int(np.argmin(losses))
        min_loss_epoch = int(epochs[min_loss_idx])
        min_loss_value = float(losses[min_loss_idx])

        sns.lineplot(ax=axes[0], x=epochs, y=losses, marker="o", color="tab:blue")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].axvline(min_loss_epoch, color="tab:green", linestyle="--", alpha=0.6, linewidth=1.2)
        axes[0].annotate(
            f"min={min_loss_value:.4f}",
            xy=(min_loss_epoch, min_loss_value),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="tab:green",
        )

        if np.all(losses > 0):
            log_losses = np.log(losses)
            sns.lineplot(ax=axes[1], x=epochs, y=log_losses, marker="o", color="tab:orange")
            axes[1].set_ylabel("Log Loss")
        else:
            axes[1].text(
                0.5,
                0.5,
                "Log loss unavailable\n(non-positive values)",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=10,
            )
            axes[1].set_ylabel("Log Loss")
        axes[1].set_title("Log Loss")
        axes[1].set_xlabel("Epoch")

        window = min(5, len(losses))
        if window >= 2:
            kernel = np.ones(window) / window
            smoothed = np.convolve(losses, kernel, mode="valid")
            smoothed_epochs = epochs[window - 1 :]
            sns.lineplot(ax=axes[2], x=smoothed_epochs, y=smoothed, marker="o", color="tab:purple")
            axes[2].set_ylabel("Rolling Mean Loss")
            axes[2].set_title(f"Rolling Mean Loss (window={window})")
            axes[2].set_xlabel("Epoch")
        else:
            axes[2].text(
                0.5,
                0.5,
                "Insufficient epochs\nfor rolling statistics",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
                fontsize=10,
            )
            axes[2].set_title("Rolling Mean Loss")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Rolling Mean Loss")

        fig.suptitle("Training Metrics", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(self.metrics_plot_path, dpi=200)
        plt.close(fig)
        logger.info("Saved training metrics plot to %s", self.metrics_plot_path)
        self._persist_run_file()

    def _should_log_epoch(self, epoch: int) -> bool:
        if epoch == 1 or epoch == self.num_epochs:
            return True
        return epoch % self._log_every_n_epochs == 0

    def _after_epoch(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Run scheduler updates and evaluate early stopping conditions."""
        loss_value = metrics.get("loss")

        if self.scheduler is not None:
            previous_lrs = [group["lr"] for group in self.optimizer.param_groups]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if loss_value is not None:
                    self.scheduler.step(loss_value)
            else:
                self.scheduler.step()
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            if current_lrs != previous_lrs:
                logger.info("Adjusted learning rates from %s to %s", previous_lrs, current_lrs)

        if self.early_stopping_patience is None or loss_value is None:
            return False

        if loss_value < (self._best_monitored_loss - self.early_stopping_min_delta):
            self._best_monitored_loss = loss_value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self.early_stopping_patience:
            self._stop_requested = True

        return self._stop_requested

    @classmethod
    def replot_metric(cls, run_file: str | Path, metric: str = "loss", *, output_path: str | Path | None = None, show: bool = False) -> Path | None:
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

        if metric == "loss":
            min_metric = min(y_values)
            if min_metric > 0:
                plt.yscale("log")
            else:
                logger.warning("Metric '%s' contains non-positive values; skipping logarithmic scaling.", metric)

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        logger.info("Saved %s metric plot to %s", metric, output)
        return output
