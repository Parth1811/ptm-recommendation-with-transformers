"""Training harness for the similarity transformer model."""

from __future__ import annotations

from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
from beautilog import logger

from config import ConfigParser, SimilarityModelConfig, SimilarityTrainerConfig
from dataloader import build_dataset_token_loader, build_model_embedding_loader
from dataloader.ranking import compute_true_ranks
from loss import ranking_loss
from model import SimilarityTransformerModel

from .base_trainer import Trainer


@dataclass
class SimilarityBatch:
    dataset_tokens: torch.Tensor
    model_embeddings: torch.Tensor
    model_indices: torch.Tensor
    true_ranks: torch.Tensor
    dataset_metadata: Dict[str, Any]
    model_metadata: Dict[str, Any]

    def to(self, device: torch.device) -> "SimilarityBatch":
        """Return a copy of the batch with tensor payloads moved to the target device."""
        return SimilarityBatch(
            dataset_tokens=self.dataset_tokens.to(device),
            model_embeddings=self.model_embeddings.to(device),
            model_indices=self.model_indices.to(device=device, dtype=torch.long),
            true_ranks=self.true_ranks.to(device=device, dtype=torch.long),
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata,
        )


class SimilarityBatchIterable:
    """Wrap dataset/model loaders to yield combined training batches."""

    def __init__(self) -> None:
        self.dataset_loader = build_dataset_token_loader()
        self.model_loader = build_model_embedding_loader()
        self._warned_missing_ranks = False

    def __len__(self) -> int:
        return len(self.dataset_loader)

    def __iter__(self) -> Iterator[SimilarityBatch]:
        model_cycle = cycle(self.model_loader)
        for dataset_batch in self.dataset_loader:
            model_batch = next(model_cycle)
            yield self._merge_batches(dataset_batch, model_batch)

    def _merge_batches(self, dataset_batch: Dict[str, Any], model_batch: Dict[str, Any]) -> SimilarityBatch:
        tokens = dataset_batch.get("dataset_tokens")
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("dataset_tokens must be a torch.Tensor")
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        elif tokens.dim() != 3:
            raise ValueError("dataset_tokens must have shape [batches, classes, dim]")

        model_embeddings = model_batch.get("model_tokens")
        if not isinstance(model_embeddings, torch.Tensor):
            raise TypeError("model_tokens must be a torch.Tensor")

        model_indices = model_batch.get("model_indices")
        if not isinstance(model_indices, torch.Tensor):
            raise TypeError("model_indices must be a torch.Tensor")

        dataset_name = dataset_batch.get("dataset_name")
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0] if dataset_name else ""
        dataset_name = str(dataset_name or "")

        model_names = model_batch.get("model_names")
        if isinstance(model_names, tuple):
            model_names = list(model_names)
        if not isinstance(model_names, list):
            model_names = list(model_names) if model_names is not None else []

        true_ranks = compute_true_ranks(dataset_name, model_names)
        if true_ranks.numel() == 0:
            if not self._warned_missing_ranks:
                logger.warning(
                    "Unable to derive rankings for dataset '%s'; using uniform ranks.",
                    dataset_name,
                )
                self._warned_missing_ranks = True
            true_ranks = torch.arange(1, model_embeddings.shape[0] + 1, dtype=torch.long)

        dataset_batch = dict(dataset_batch)
        dataset_batch["true_ranks"] = true_ranks

        return SimilarityBatch(
            dataset_tokens=tokens,
            model_embeddings=model_embeddings,
            model_indices=model_indices,
            true_ranks=true_ranks,
            dataset_metadata=deepcopy(dataset_batch),
            model_metadata=deepcopy(model_batch),
        )


class SimilarityTransformerTrainer(Trainer):
    """Trainer that optimizes the similarity transformer with listwise ranking loss."""

    def __init__(self) -> None:
        training_config = ConfigParser.get(SimilarityTrainerConfig)

        model = SimilarityTransformerModel()
        self.batch_iterable = SimilarityBatchIterable()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        super().__init__(
            model=model,
            dataloader=self.batch_iterable,
            optimizer=optimizer,
            config=training_config,
        )

        self.model.to(self.device)

        self.ranking_loss_weight = float(self.config.ranking_loss_weight)
        self.logit_l2_weight = float(self.config.logit_l2_weight)
        self.extra_loss_weight = float(self.config.extra_loss_weight)
        self._last_batch_metrics: Dict[str, float] = {}

    def _train_batch(
        self, batch: SimilarityBatch
    ) -> tuple[float, int, float | None, Dict[str, float]]:  # type: ignore[override]
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics, batch_size = self._forward_batch(batch)
        loss.backward()
        grad_norm = self._apply_gradient_clipping()
        self.optimizer.step()
        self._last_batch_metrics = metrics
        return float(loss.detach().item()), batch_size, grad_norm, metrics

    def _forward_batch(
        self, batch: SimilarityBatch
    ) -> tuple[torch.Tensor, Dict[str, float], int]:
        batch = batch.to(self.device)

        model_embeddings = batch.model_embeddings
        dataset_tokens = batch.dataset_tokens
        model_indices = batch.model_indices
        true_ranks = batch.true_ranks

        num_models = model_embeddings.size(0)

        logits = self.model(model_embeddings, dataset_tokens)
        if logits.dim() != 2:
            raise ValueError("Model must return logits with shape [batch, num_models].")

        model_indices = model_indices.clamp(max=logits.size(1) - 1)
        pred_scores = logits.gather(1, model_indices.unsqueeze(1)).squeeze(1)

        rank_loss = ranking_loss(pred_scores, true_ranks)
        loss = self.ranking_loss_weight * rank_loss

        logit_penalty = logits.pow(2).mean()
        if self.logit_l2_weight > 0.0:
            loss = loss + self.logit_l2_weight * logit_penalty

        metrics: Dict[str, float] = {
            "ranking_loss": float(rank_loss.detach().item()),
            "logit_l2": float(logit_penalty.detach().item()),
        }

        if self.extra_loss_weight != 0.0:
            metrics["extra_loss"] = 0.0  # placeholder for future extension

        batch_size = int(num_models)
        return loss, metrics, batch_size

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("compute_loss is unused; training logic is handled in _forward_batch.")

    def _collect_batch_metrics(self) -> Dict[str, float]:  # type: ignore[override]
        return dict(self._last_batch_metrics)

    def on_train_begin(self) -> None:  # type: ignore[override]
        logger.info("Starting similarity transformer training with %d dataset batches.", len(self.batch_iterable))

    def on_train_end(self) -> None:  # type: ignore[override]
        if not self.history:
            return

        final_metrics = self.history[-1]["metrics"]
        final_loss = final_metrics.get("loss")
        if final_loss is None:
            logger.warning("Final loss missing; skipping similarity transformer checkpoint save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.config.model_save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"similarity_transformer_weights.loss_{final_loss:.6f}.{timestamp}.pt"
        torch.save(self.model.state_dict(), save_path)
        logger.info("Similarity transformer weights saved to %s", save_path)
        logger.info("Final training metrics: %s", final_metrics)
