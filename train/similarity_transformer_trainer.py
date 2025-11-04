"""Training harness for the similarity transformer model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Iterator

import torch
from beautilog import logger

from config import (ConfigParser, DatasetTokenLoaderConfig,
                    ModelEmbeddingLoaderConfig, SimilarityModelConfig,
                    SimilarityTrainerConfig)
from dataloader import build_dataset_token_loader, build_model_embedding_loader
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


class SimilarityBatchIterable:
    """Wrap dataset/model loaders to yield combined training batches."""

    def __init__(self) -> None:
        self.dataset_loader = build_dataset_token_loader()
        self.model_loader = build_model_embedding_loader()

    def __len__(self) -> int:
        return len(self.dataset_loader)

    def __iter__(self) -> Iterator[SimilarityBatch]:
        model_cycle = cycle(self.model_loader)
        for dataset_batch in self.dataset_loader:
            model_batch = next(model_cycle)
            yield self._merge_batches(dataset_batch, model_batch)

    def _merge_batches(self, dataset_batch: Dict[str, Any], model_batch: Dict[str, Any]) -> SimilarityBatch:
        tokens = dataset_batch.get("dataset_tokens")

        if tokens.dim() != 3:
            raise ValueError("dataset_tokens must have shape [batches, classes, dim]")

        model_embeddings = model_batch.get("model_tokens")
        model_indices = model_batch.get("model_indices")

        true_ranks = dataset_batch.get("true_ranks")
        if true_ranks is None:
            true_ranks = torch.tensor(model_indices, dtype=torch.float32)

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
        ConfigParser.load()
        self.trainer_cfg = ConfigParser.get(SimilarityTrainerConfig)
        self.model_cfg = ConfigParser.get(SimilarityModelConfig)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SimilarityTransformerModel(self.model_cfg)
        model.to(self.device)

        if getattr(self.trainer_cfg, "grad_clip", None) is not None:
            setattr(self.trainer_cfg, "gradient_clip_norm", self.trainer_cfg.grad_clip)

        self.batch_iterable = SimilarityBatchIterable()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.trainer_cfg.learning_rate,
            weight_decay=self.trainer_cfg.weight_decay,
        )

        super().__init__(
            model=model,
            dataloader=self.batch_iterable,
            optimizer=optimizer,
            config=self.trainer_cfg,
        )

        self.ranking_loss_weight = float(self.trainer_cfg.ranking_loss_weight)
        self.logit_l2_weight = float(self.trainer_cfg.logit_l2_weight)
        self.extra_loss_weight = float(self.trainer_cfg.extra_loss_weight)
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
        model_embeddings = batch.model_embeddings.to(self.device)
        dataset_tokens = batch.dataset_tokens.to(self.device)
        model_indices = batch.model_indices.to(self.device)
        true_ranks = batch.true_ranks.to(self.device).to(torch.long)

        num_models = model_embeddings.size(0)

        if dataset_tokens.size(0) == 1:
            dataset_tokens = dataset_tokens.expand(num_models, -1, -1)
        elif dataset_tokens.size(0) != num_models:
            dataset_tokens = dataset_tokens.repeat_interleave(num_models, dim=0)

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
