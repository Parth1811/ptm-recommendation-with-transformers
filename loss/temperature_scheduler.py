"""Temperature scheduler for ranking losses."""

from __future__ import annotations

import math
from typing import Literal


class TemperatureScheduler:
    """Scheduler for temperature annealing in ranking losses.

    Starts with high temperature (softer distributions) for better initial gradients,
    then gradually decreases to lower temperature (sharper distributions) as training progresses.

    Example:
        >>> scheduler = TemperatureScheduler(
        ...     initial_temp=3.0,
        ...     final_temp=1.0,
        ...     total_steps=1000,
        ...     schedule="cosine"
        ... )
        >>> for step in range(1000):
        ...     temp = scheduler.get_temperature(step)
        ...     loss = ranking_loss(logits, ranks, temperature=temp)
    """

    def __init__(
        self,
        initial_temp: float = 3.0,
        final_temp: float = 1.0,
        total_steps: int = 1000,
        schedule: Literal["linear", "exponential", "cosine"] = "cosine",
        warmup_steps: int = 0,
    ):
        """Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature (higher = softer distributions).
            final_temp: Ending temperature (typically 1.0 for no scaling).
            total_steps: Total number of training steps (epochs * batches_per_epoch).
            schedule: Decay schedule type:
                - "linear": Linear decay from initial to final
                - "exponential": Exponential decay
                - "cosine": Cosine annealing (smooth decay, recommended)
            warmup_steps: Number of steps to keep initial_temp before starting decay.
        """
        if initial_temp < final_temp:
            raise ValueError(f"initial_temp ({initial_temp}) must be >= final_temp ({final_temp})")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_temperature(self, step: int | None = None) -> float:
        """Get temperature for current training step.

        Args:
            step: Training step (if None, uses internal counter).

        Returns:
            Current temperature value.
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        # Warmup phase: keep initial temperature
        if step < self.warmup_steps:
            return self.initial_temp

        # Adjust step for warmup
        adjusted_step = step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps

        # Clamp to valid range
        progress = min(adjusted_step / max(adjusted_total, 1), 1.0)

        # Compute temperature based on schedule
        if self.schedule == "linear":
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.schedule == "exponential":
            # Exponential decay: T = T_init * (T_final / T_init) ^ progress
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.schedule == "cosine":
            # Cosine annealing: smooth decay
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return temp

    def step(self) -> float:
        """Increment step counter and return current temperature.

        Returns:
            Current temperature value.
        """
        return self.get_temperature()

    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_step = 0

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing."""
        return {
            "initial_temp": self.initial_temp,
            "final_temp": self.final_temp,
            "total_steps": self.total_steps,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.initial_temp = state_dict["initial_temp"]
        self.final_temp = state_dict["final_temp"]
        self.total_steps = state_dict["total_steps"]
        self.schedule = state_dict["schedule"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.current_step = state_dict["current_step"]


class AdaptiveTemperatureScheduler:
    """Adaptive temperature scheduler based on prediction entropy.

    Adjusts temperature based on how confident the model's predictions are:
    - If predictions are too uniform (high entropy), increase temperature
    - If predictions are too peaked (low entropy), decrease temperature

    This helps maintain optimal gradient flow throughout training.
    """

    def __init__(
        self,
        initial_temp: float = 2.0,
        min_temp: float = 0.5,
        max_temp: float = 5.0,
        target_entropy: float | None = None,
        adaptation_rate: float = 0.1,
    ):
        """Initialize adaptive temperature scheduler.

        Args:
            initial_temp: Starting temperature.
            min_temp: Minimum allowed temperature.
            max_temp: Maximum allowed temperature.
            target_entropy: Target entropy for predictions (if None, computed as log(num_classes)).
            adaptation_rate: How quickly to adapt temperature (0-1, higher = faster adaptation).
        """
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.target_entropy = target_entropy
        self.adaptation_rate = adaptation_rate

    def update(self, predictions: "torch.Tensor") -> float:
        """Update temperature based on prediction entropy.

        Args:
            predictions: Model predictions (logits or probabilities), shape [B, N].

        Returns:
            Updated temperature value.
        """
        import torch

        # Compute softmax probabilities if not already
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            probs = torch.softmax(predictions, dim=-1)
        else:
            probs = predictions

        # Compute entropy: H = -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

        # Set target entropy if not specified (uniform distribution entropy)
        if self.target_entropy is None:
            num_classes = predictions.shape[-1]
            self.target_entropy = math.log(num_classes)

        # Adapt temperature based on entropy
        # If entropy > target: predictions too uniform, increase temperature
        # If entropy < target: predictions too peaked, decrease temperature
        entropy_diff = entropy - self.target_entropy
        temp_adjustment = self.adaptation_rate * entropy_diff

        # Update temperature
        self.current_temp = max(
            self.min_temp,
            min(self.max_temp, self.current_temp + temp_adjustment)
        )

        return self.current_temp

    def get_temperature(self) -> float:
        """Get current temperature without updating."""
        return self.current_temp

    def reset(self, initial_temp: float | None = None):
        """Reset temperature to initial value."""
        if initial_temp is not None:
            self.current_temp = initial_temp
