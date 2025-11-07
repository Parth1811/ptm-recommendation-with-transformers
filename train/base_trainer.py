"""Base training utilities."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from beautilog import logger
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import BaseTrainerConfig


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
        val_dataloader: DataLoader | Iterable[torch.Tensor] | None = None,
    ) -> None:
        if getattr(config, "num_epochs", None) is None:
            raise ValueError("Training configuration must provide a num_epochs attribute.")

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_epochs = config.num_epochs
        self.history: list[dict[str, Any]] = []
        self.gradient_clip_norm = getattr(config, "gradient_clip_norm", None)
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)

        self._monitor_metric_name = getattr(config, "scheduler_monitor", "validation.loss")
        self.validation_interval_epochs = max(1, int(getattr(config, "validation_interval_epochs", 1)))
        self.log_every_n_steps = max(1, int(getattr(config, "log_every_n_steps", 1)))
        self._log_grad_stats = bool(getattr(config, "log_grad_stats", False))
        self._global_step = 0

        self.early_stopping_patience = getattr(config, "early_stopping_patience", None)
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            self.early_stopping_patience = None
        self.early_stopping_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
        self._best_monitored_metric = float("inf")
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
        self.seed = getattr(config, "seed", None)
        if self.seed is not None:
            self._set_seed(int(self.seed))
        logger.info("Initialized trainer %s at %s", self.run_name, self.run_file_path)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Set random seed to %d for reproducibility.", seed)


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
            train_metrics = self._train_epoch(epoch, progress_bar, total_batches)
            val_metrics = None
            if self._should_validate(epoch):
                val_metrics = self._validate_epoch(epoch)
            self.record_data(epoch, train_metrics, val_metrics)
            stop_training = self._after_epoch(epoch, train_metrics, val_metrics)
            if self._should_log_epoch(epoch):
                if val_metrics is not None:
                    logger.info(
                        "Epoch %d metrics: train=%s | validation=%s",
                        epoch,
                        train_metrics,
                        val_metrics,
                    )
                else:
                    logger.info("Epoch %d metrics: %s", epoch, train_metrics)
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

    def _forward_batch(self, batch: Any) -> torch.Tensor:
        """Forward pass for a batch, returning the computed loss.

        This is the abstract method that child trainers can override to handle
        complex batch structures and multiple tensor inputs. The default implementation
        calls compute_loss(), which works for simple tensor batches and dict/dataclass
        batches where compute_loss() handles all unpacking and device movement.

        For trainers with very complex batch handling (e.g., SimilarityTransformerTrainer),
        override this method to:
        1. Move batch to device (handle dataclass .to(), dict values, etc.)
        2. Call model with unpacked tensors
        3. Compute loss from model outputs
        4. Return scalar loss tensor

        Args:
            batch: A batch object in any format (Tensor, dict, dataclass, etc.)

        Returns:
            A scalar torch.Tensor representing the loss.

        Default behavior:
            Calls self.compute_loss(batch) which must be implemented by child classes.

        Example for complex batch (override _forward_batch instead of compute_loss):
            def _forward_batch(self, batch: SimilarityBatch) -> torch.Tensor:
                batch = batch.to(self.device)
                output = self.model(batch.embeddings, batch.tokens)
                loss = self.compute_loss_from_output(output, batch.targets)
                return loss
        """
        return self.compute_loss(batch)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch.

        Subclasses must implement this method to return a scalar loss tensor.

        IMPORTANT: This method is called by _forward_batch(). If your trainer needs
        complex batch handling (multiple tensor inputs, dataclass unpacking, etc.),
        you have two options:

        Option 1: Keep using compute_loss() if your batch structure allows
        --------
        For trainers where batch unpacking is straightforward (dicts with known keys,
        dataclasses with .to() method), implement compute_loss() to:
        1. Unpack the batch components from the container (dict keys, dataclass fields)
        2. Move individual tensors to self.device as needed
        3. Call self.model() with the unpacked tensors
        4. Compute and return scalar loss

        For simple tensor batches, the batch will already be on the correct device.

        Option 2: Override _forward_batch() for advanced use cases
        ---------
        For trainers with highly complex batch handling (multiple iterables cycling,
        structured outputs that need metric collection, etc.), override _forward_batch()
        instead. This gives you full control over batch processing while keeping the
        training loop infrastructure intact.

        Args:
            batch: A batch object. Can be:
                - torch.Tensor: Already on self.device
                - dict: Contains tensor values that may need device movement
                - Custom dataclass: May need custom unpacking and device handling

        Returns:
            A scalar torch.Tensor representing the loss.

        Example 1 - dict batch (implement compute_loss):
            def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                tokens = batch["tokens"].to(self.device)
                targets = batch["targets"].to(self.device)
                logits = self.model(tokens)
                return F.cross_entropy(logits, targets)

        Example 2 - dataclass batch with .to() method (implement compute_loss):
            def compute_loss(self, batch: MyBatch) -> torch.Tensor:
                batch = batch.to(self.device)
                logits = self.model(batch.data, batch.metadata)
                return F.mse_loss(logits, batch.targets)

        Example 3 - very complex batch (override _forward_batch):
            def _forward_batch(self, batch: ComplexBatch) -> torch.Tensor:
                # Custom logic here
                batch = batch.to(self.device)
                output = self.model(batch.data1, batch.data2, batch.data3)
                loss = F.some_loss(output, batch.targets)
                return loss
        """
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

            self._global_step += 1
            current_lrs = self._get_current_lrs()
            lr_display = ", ".join(f"{lr:.6g}" for lr in current_lrs)
            grad_display = f"{grad_norm:.4f}" if grad_norm is not None else "nan"
            metric_display = ""
            if batch_metrics:
                preview_items = list(batch_metrics.items())[:3]
                metric_display = " | ".join(f"{key}={value:.4f}" for key, value in preview_items)
            log_message = (
                f"Step {self._global_step} | epoch={epoch} | batch={batch_index} | "
                f"lr={lr_display} | grad_norm={grad_display} | batch_loss={loss_value:.4f}"
            )
            if metric_display:
                log_message = f"{log_message} | {metric_display}"
            if self._log_grad_stats or (self._global_step % self.log_every_n_steps == 0):
                logger.info(log_message)

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

    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        running_loss = 0.0
        samples_seen = 0
        aggregated_metrics: dict[str, float] = {}
        aggregated_counts: dict[str, int] = {}

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss_value, batch_size, batch_metrics = self._evaluate_batch(batch)
                running_loss += loss_value * batch_size
                samples_seen += batch_size
                for key, value in batch_metrics.items():
                    aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
                    aggregated_counts[key] = aggregated_counts.get(key, 0) + 1

        average_loss = running_loss / max(samples_seen, 1)
        metrics: dict[str, float] = {"loss": float(average_loss)}
        for key, total in aggregated_metrics.items():
            metrics[key] = float(total / max(aggregated_counts.get(key, 1), 1))
        return metrics

    def _train_batch(self, batch: torch.Tensor) -> tuple[float, int, float | None, dict[str, float]]:
        """Execute a single training batch with loss computation and gradient updates.

        This method coordinates the training loop:
        1. Extracts batch size from batch via _get_batch_size()
        2. Performs forward pass via _forward_batch() (which can be overridden by subclasses)
        3. Applies backward pass and optimizer step
        4. Collects metrics
        5. Returns loss, batch_size, gradient norm, and batch metrics

        The forward pass is delegated to _forward_batch() which:
        - By default calls compute_loss() for simple cases
        - Can be overridden for complex batch handling (e.g., SimilarityTransformerTrainer)
        """
        batch_size = self._get_batch_size(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward_batch(batch)
        loss.backward()
        grad_norm = self._apply_gradient_clipping()
        self.optimizer.step()
        batch_metrics = self._collect_batch_metrics()
        return loss.detach().item(), batch_size, grad_norm, batch_metrics

    def _evaluate_batch(self, batch: torch.Tensor) -> tuple[float, int, dict[str, float]]:
        """Execute a single evaluation batch without gradient updates.

        This method performs validation/evaluation:
        1. Extracts batch size via _get_batch_size()
        2. Performs forward pass via _forward_batch() (which can be overridden by subclasses)
        3. Collects metrics
        4. Returns loss, batch_size, and batch metrics

        Note: No backward pass or optimizer step occurs during evaluation.
        The forward pass is delegated to _forward_batch() (same as training).
        """
        batch_size = self._get_batch_size(batch)
        loss = self._forward_batch(batch)
        batch_metrics = self._collect_batch_metrics()
        return loss.detach().item(), batch_size, batch_metrics

    def _get_current_lrs(self) -> list[float]:
        return [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]

    def _resolve_monitor_value(
        self,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float] | None,
    ) -> float | None:
        spec = (self._monitor_metric_name or "validation.loss").strip().lower()

        def _get(mapping: Mapping[str, float] | None, key: str) -> float | None:
            if mapping is None:
                return None
            return mapping.get(key)

        if spec in {"loss", "train_loss", "train.loss"}:
            return _get(train_metrics, "loss")
        if spec in {"val_loss", "validation_loss", "validation.loss"}:
            value = _get(val_metrics, "loss")
            if value is not None:
                return value
            return _get(train_metrics, "loss")

        if "." in spec:
            prefix, metric_name = spec.split(".", 1)
            if prefix in {"train", "training"}:
                return _get(train_metrics, metric_name)
            if prefix in {"val", "validation"}:
                value = _get(val_metrics, metric_name)
                if value is not None:
                    return value
                return _get(train_metrics, metric_name)

        value = _get(train_metrics, spec)
        if value is not None:
            return value
        return _get(val_metrics, spec)

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

    def _get_batch_size(self, batch: Any) -> int:
        """Extract the batch size from a batch of any format.

        This method handles extracting batch size from:
        - torch.Tensor: Returns batch.size(0)
        - dict: Returns size of first tensor value's first dimension
        - dataclass with .to() method: Assumes first tensor field has batch size in dim 0
        - custom objects: Subclasses can override for their specific batch type

        Args:
            batch: A batch object of any format. Can be Tensor, dict, dataclass, or custom object.

        Returns:
            The batch size (number of samples in the batch).

        Raises:
            ValueError: If batch size cannot be determined from the batch format.
        """
        # Handle torch.Tensor batches (simple case)
        if isinstance(batch, torch.Tensor):
            return batch.size(0)

        # Handle dictionary batches
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    return value.size(0)
            raise ValueError(f"No tensor values found in batch dictionary with keys: {list(batch.keys())}")

        # Handle dataclass batches with .to() method (like SimilarityBatch)
        if hasattr(batch, "to") and callable(getattr(batch, "to")):
            # Try to find a tensor attribute to extract batch size from
            for attr_name in dir(batch):
                if not attr_name.startswith("_"):
                    attr_value = getattr(batch, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        return attr_value.size(0)
            raise ValueError(f"No tensor attributes found in batch object {batch.__class__.__name__}")

        raise ValueError(
            f"Unable to extract batch size from batch of type {type(batch).__name__}. "
            f"Batch must be torch.Tensor, dict, or object with tensor attributes."
        )

    def record_data(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """Persist epoch metrics and emit structured logs."""
        metrics_payload: dict[str, Any] = {"train": train_metrics}
        if "loss" in train_metrics:
            metrics_payload["loss"] = float(train_metrics["loss"])
        if val_metrics:
            metrics_payload["validation"] = val_metrics
            if "loss" in val_metrics:
                metrics_payload["val_loss"] = float(val_metrics["loss"])

        record = {
            "epoch": epoch,
            "metrics": metrics_payload,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.history.append(record)
        self._persist_run_file()

    def _save_metric_plots(self) -> None:
        if not self.history:
            return

        epochs = np.array([entry["epoch"] for entry in self.history], dtype=float)
        losses = np.array([entry["metrics"].get("loss") for entry in self.history], dtype=float)
        val_loss_values = [entry["metrics"].get("val_loss") for entry in self.history]
        val_losses = np.array(
            [value if value is not None else np.nan for value in val_loss_values],
            dtype=float,
        )

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        axes = np.atleast_1d(axes)

        min_loss_idx = int(np.argmin(losses))
        min_loss_epoch = int(epochs[min_loss_idx])
        min_loss_value = float(losses[min_loss_idx])

        sns.lineplot(ax=axes[0], x=epochs, y=losses, color="tab:blue", label="Train")
        val_mask = np.isfinite(val_losses)
        if val_mask.any():
            sns.lineplot(
                ax=axes[0],
                x=epochs[val_mask],
                y=val_losses[val_mask],
                color="tab:orange",
                label="Validation",
            )
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
        if val_mask.any():
            axes[0].legend()

        if np.all(losses > 0):
            log_losses = np.log(losses)
            sns.lineplot(ax=axes[1], x=epochs, y=log_losses, color="tab:orange", label="Train")
            if val_mask.any() and np.all(val_losses[val_mask] > 0):
                sns.lineplot(
                    ax=axes[1],
                    x=epochs[val_mask],
                    y=np.log(val_losses[val_mask]),
                    color="tab:red",
                    label="Validation",
                )
            axes[1].set_ylabel("Log Loss")
            if val_mask.any():
                axes[1].legend()
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
            sns.lineplot(ax=axes[2], x=smoothed_epochs, y=smoothed, color="tab:purple")
            if val_mask.sum() >= window:
                val_epochs = epochs[val_mask]
                val_series = val_losses[val_mask]
                val_kernel = np.ones(window) / window
                val_smoothed = np.convolve(val_series, val_kernel, mode="valid")
                val_smoothed_epochs = val_epochs[window - 1 :]
                sns.lineplot(ax=axes[2], x=val_smoothed_epochs, y=val_smoothed, color="tab:brown")
            axes[2].set_ylabel("Rolling Mean Loss")
            axes[2].set_title(f"Rolling Mean Loss (window={window})")
            axes[2].set_xlabel("Epoch")
            if val_mask.sum() >= window:
                axes[2].legend(["Train", "Validation"])
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

    def _should_validate(self, epoch: int) -> bool:
        if self.val_dataloader is None:
            return False
        if epoch % self.validation_interval_epochs == 0:
            return True
        if epoch == self.num_epochs:
            return True
        return False

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> bool:
        """Run scheduler updates and evaluate early stopping conditions."""
        monitor_value = self._resolve_monitor_value(train_metrics, val_metrics)

        if self.scheduler is not None:
            previous_lrs = [group["lr"] for group in self.optimizer.param_groups]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
                else:
                    logger.warning(
                        "Scheduler monitor '%s' unavailable at epoch %d; skipping scheduler step.",
                        self._monitor_metric_name,
                        epoch,
                    )
            else:
                self.scheduler.step()
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            if current_lrs != previous_lrs:
                logger.info("Adjusted learning rates from %s to %s", previous_lrs, current_lrs)

        if self.early_stopping_patience is None or monitor_value is None:
            return False

        if monitor_value < (self._best_monitored_metric - self.early_stopping_min_delta):
            self._best_monitored_metric = monitor_value
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
            value = metrics.get(metric)
            if value is None and "." in metric:
                prefix, child_key = metric.split(".", 1)
                nested = metrics.get(prefix)
                if isinstance(nested, Mapping):
                    value = nested.get(child_key)
            if value is None:
                continue
            x_values.append(entry["epoch"])
            y_values.append(value)

        if not x_values:
            raise KeyError(f"Metric '{metric}' not present in run history.")


        plt.figure(figsize=(8, 4.5))
        # plt.plot(x_values, y_values, marker="o")
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
