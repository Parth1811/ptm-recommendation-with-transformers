"""Training harness for the similarity transformer model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import torch
import torch.nn.functional as F
from beautilog import logger
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ConfigParser, SimilarityModelConfig, SimilarityTrainerConfig
from dataloader import build_dataset_token_loader, build_model_embedding_loader
from dataloader.ranking import compute_true_ranks
from model import SimilarityTransformerModel

from loss import ranking_loss
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

    def __init__(
        self,
        *,
        splits: Sequence[str] | None = None,
        shuffle: bool | None = None,
        max_batches: int | None = None,
        max_models: int | None = None,
        dataset_names: Sequence[str] | None = None,
    ) -> None:
        dataset_kwargs: dict[str, Any] = {}
        if splits is not None:
            dataset_kwargs["splits"] = splits
        if shuffle is not None:
            dataset_kwargs["shuffle"] = shuffle
        if dataset_names is not None:
            dataset_kwargs["dataset_names"] = dataset_names
        if max_batches is not None:
            dataset_kwargs["max_shards"] = max_batches

        model_kwargs: dict[str, Any] = {}
        if shuffle is not None:
            model_kwargs["shuffle"] = shuffle
        if max_models is not None:
            model_kwargs["max_models"] = max_models

        self.dataset_loader = build_dataset_token_loader(**dataset_kwargs)
        self.model_loader = build_model_embedding_loader(**model_kwargs)
        self._warned_missing_ranks = False
        self.max_batches = max_batches

    def __len__(self) -> int:
        base_length = len(self.dataset_loader)
        if self.max_batches is not None:
            return min(base_length, self.max_batches)
        return base_length

    def __iter__(self) -> Iterator[SimilarityBatch]:
        model_cycle = cycle(self.model_loader)
        for batch_index, dataset_batch in enumerate(self.dataset_loader, start=1):
            if self.max_batches is not None and batch_index > self.max_batches:
                break
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
            dataset_metadata=copy.deepcopy(dataset_batch),
            model_metadata=copy.deepcopy(model_batch),
        )


class SimilarityTransformerTrainer(Trainer):
    """Trainer that optimizes the similarity transformer with temperature-scaled InfoNCE loss."""

    def __init__(self) -> None:
        training_config = ConfigParser.get(SimilarityTrainerConfig)
        model_config = ConfigParser.get(SimilarityModelConfig)

        model = SimilarityTransformerModel()

        train_iterable = SimilarityBatchIterable(
            splits=training_config.train_splits,
            shuffle=training_config.shuffle,
            max_models=model_config.num_models,
        )
        val_iterable = None
        if training_config.validation_splits:
            val_iterable = SimilarityBatchIterable(
                splits=training_config.validation_splits,
                shuffle=False,
                max_models=model_config.num_models,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=training_config.scheduler_mode,
            factor=training_config.scheduler_factor,
            patience=training_config.scheduler_patience,
            cooldown=training_config.scheduler_cooldown,
            threshold=training_config.scheduler_threshold,
            threshold_mode=training_config.scheduler_threshold_mode,
            min_lr=training_config.scheduler_min_lr,
        )

        super().__init__(
            model=model,
            dataloader=train_iterable,
            optimizer=optimizer,
            config=training_config,
            scheduler=scheduler,
            val_dataloader=val_iterable,
        )

        self.model.to(self.device)

        self.train_iterable = train_iterable
        self.val_iterable = val_iterable
        self.label_smoothing = float(training_config.label_smoothing)
        self.ranking_loss_weight = float(max(training_config.ranking_loss_weight, 0.0))
        self.hard_negative_top_k = int(max(training_config.hard_negative_top_k, 0))
        self.hard_negative_margin = float(max(training_config.hard_negative_margin, 0.0))
        self.hard_negative_weight = float(max(training_config.hard_negative_weight, 0.0))
        self.overfit_subset_size = (
            int(training_config.overfit_subset_size)
            if training_config.overfit_subset_size is not None
            else None
        )
        self.overfit_max_epochs = max(int(training_config.overfit_max_epochs), 0)
        self.overfit_shuffle = bool(getattr(training_config, "overfit_shuffle", True))
        self.enable_overfit_check = bool(getattr(training_config, "enable_overfit_check", True))
        self.log_activation_stats = bool(getattr(training_config, "log_activation_stats", False))
        self.logit_l2_weight = float(training_config.logit_l2_weight)
        self._last_batch_metrics: Dict[str, float] = {}

    def train(self) -> list[dict[str, Any]]:  # type: ignore[override]
        if self.enable_overfit_check and self.overfit_subset_size:
            self._run_overfit_sanity_check()
        return super().train()

    def _train_batch(
        self, batch: SimilarityBatch
    ) -> tuple[float, int, float | None, Dict[str, float]]:  # type: ignore[override]
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics, batch_size = self._forward_batch(batch, training=True)
        loss.backward()
        grad_norm = self._apply_gradient_clipping()
        self.optimizer.step()
        self._last_batch_metrics = metrics
        return float(loss.detach().item()), batch_size, grad_norm, metrics

    def _evaluate_batch(self, batch: SimilarityBatch) -> tuple[float, int, Dict[str, float]]:  # type: ignore[override]
        loss, metrics, batch_size = self._forward_batch(batch, training=False)
        return float(loss.detach().item()), batch_size, metrics

    def _forward_batch(
        self,
        batch: SimilarityBatch,
        *,
        training: bool,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float], int]:
        batch = batch.to(self.device)

        dataset_tokens = F.normalize(batch.dataset_tokens, dim=-1)
        model_embeddings = F.normalize(batch.model_embeddings, dim=-1)
        output = self.model(model_embeddings, dataset_tokens)

        logits = output["logits"]
        similarities = output["similarities"]
        dataset_repr = output["dataset_repr"]
        model_repr = output["model_repr"]
        temperature = output["temperature"]

        batch_size = dataset_tokens.size(0)
        num_models = logits.size(1)
        true_ranks = batch.true_ranks.to(self.device)

        positive_indices = torch.nonzero(self._positive_mask(true_ranks), as_tuple=False).flatten()
        if positive_indices.numel() == 0:
            primary_index = 0
        else:
            primary_index = int(positive_indices[0].item())
        targets = torch.full((batch_size,), primary_index, dtype=torch.long, device=logits.device)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=float(self.label_smoothing) if self.label_smoothing > 0 else 0.0,
        )
        loss = ce_loss

        log_probs = torch.log_softmax(logits, dim=1)
        target_distribution = self._build_target_distribution(
            true_ranks,
            num_models,
            dtype=logits.dtype,
            device=logits.device,
        )
        target_distribution = target_distribution.unsqueeze(0).expand(batch_size, -1)
        info_nce_loss = -(target_distribution * log_probs).sum(dim=1).mean()

        metrics: Dict[str, float] = {}
        if collect_metrics:
            metrics["ce_loss"] = float(ce_loss.detach().item())
            metrics["info_nce"] = float(info_nce_loss.detach().item())

        if self.ranking_loss_weight > 0:
            per_instance_losses = []
            for instance_logits in logits:
                per_instance_losses.append(
                    ranking_loss(instance_logits, true_ranks.float(), reverse_order=False)
                )
            if per_instance_losses:
                rank_loss = torch.stack(per_instance_losses).mean()
                loss = loss + self.ranking_loss_weight * rank_loss
                if collect_metrics:
                    metrics["ranking_loss"] = float(rank_loss.detach().item())

        if self.hard_negative_weight > 0:
            hard_negative_loss = self._compute_hard_negative_loss(logits, true_ranks)
            if hard_negative_loss is not None:
                loss = loss + self.hard_negative_weight * hard_negative_loss
                if collect_metrics:
                    metrics["hard_negative_loss"] = float(hard_negative_loss.detach().item())

        if self.logit_l2_weight > 0:
            logit_penalty = logits.pow(2).mean()
            loss = loss + self.logit_l2_weight * logit_penalty
            if collect_metrics:
                metrics["logit_l2"] = float(logit_penalty.detach().item())

        if collect_metrics:
            with torch.no_grad():
                probabilities = torch.softmax(logits, dim=1)
                positive_mask = self._positive_mask(true_ranks)
                positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()
                top1 = logits.argmax(dim=1, keepdim=True)
                correct = (top1 == positive_indices).any(dim=1).float().mean()

                entropy = (-probabilities * log_probs).sum(dim=1).mean()

                metrics.update(
                    {
                        "temperature": float(temperature.detach().item()),
                        "logits_mean": float(logits.mean().detach().item()),
                        "logits_std": float(logits.std(unbiased=False).detach().item()),
                        "similarity_mean": float(similarities.mean().detach().item()),
                        "similarity_std": float(similarities.std(unbiased=False).detach().item()),
                        "prob_entropy": float(entropy.detach().item()),
                        "top1_accuracy": float(correct.detach().item()),
                    }
                )

                if self.log_activation_stats:
                    metrics["dataset_repr_norm"] = float(dataset_repr.norm(dim=-1).mean().detach().item())
                    metrics["model_repr_norm"] = float(model_repr.norm(dim=-1).mean().detach().item())

        if collect_metrics:
            self._last_batch_metrics = metrics

        return loss, metrics, batch_size

    def _collect_batch_metrics(self) -> Dict[str, float]:  # type: ignore[override]
        return dict(self._last_batch_metrics)

    def _build_target_distribution(
        self,
        true_ranks: torch.Tensor,
        num_models: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        smoothing = float(self.label_smoothing)
        smoothing = max(0.0, min(smoothing, 1.0))

        positives = self._positive_mask(true_ranks)
        positive_count = int(positives.sum().item())
        target = torch.zeros(num_models, dtype=dtype, device=device)

        if positive_count == 0:
            target.fill_(1.0 / num_models)
            return target

        negative_count = num_models - positive_count
        if negative_count < 0:
            negative_count = 0

        if smoothing >= 1.0 or negative_count == 0:
            target[positives] = 1.0 / positive_count
            target = target / target.sum()
            return target

        positive_value = (1.0 - smoothing) / positive_count
        target[positives] = positive_value

        if negative_count > 0:
            negative_value = smoothing / negative_count
            target[~positives] = negative_value

        target = target / target.sum()
        return target

    def _positive_mask(self, true_ranks: torch.Tensor) -> torch.Tensor:
        min_rank = true_ranks.min()
        return true_ranks == min_rank

    def _compute_hard_negative_loss(
        self,
        logits: torch.Tensor,
        true_ranks: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.hard_negative_top_k <= 0 or logits.size(1) <= 1:
            return None

        positive_mask = self._positive_mask(true_ranks)
        negative_mask = ~positive_mask
        negative_indices = torch.nonzero(negative_mask, as_tuple=False).flatten()
        if negative_indices.numel() == 0:
            return None

        sorted_negatives = negative_indices[torch.argsort(true_ranks[negative_indices])]
        top_k = sorted_negatives[: min(self.hard_negative_top_k, sorted_negatives.numel())]
        if top_k.numel() == 0:
            return None

        positive_scores = logits[:, positive_mask]
        if positive_scores.dim() == 1:
            positive_scores = positive_scores.unsqueeze(1)
        positive_scores = positive_scores.mean(dim=1, keepdim=True)

        negative_scores = logits[:, top_k]
        margin = torch.as_tensor(
            self.hard_negative_margin,
            dtype=logits.dtype,
            device=logits.device,
        )
        hard_loss = torch.relu(margin + negative_scores - positive_scores).mean()
        return hard_loss

    def _sample_overfit_batches(self, subset_size: int) -> list[SimilarityBatch]:
        iterable = SimilarityBatchIterable(
            splits=self.config.train_splits,
            shuffle=self.overfit_shuffle,
            max_batches=subset_size,
        )
        return [batch for batch in islice(iter(iterable), subset_size)]

    def _run_overfit_sanity_check(self) -> None:
        subset_size = self.overfit_subset_size or 0
        if subset_size <= 0:
            return

        batches = self._sample_overfit_batches(subset_size)
        if not batches:
            logger.warning("Overfit sanity check skipped: unable to sample subset of size %d.", subset_size)
            return

        logger.info(
            "Running overfit sanity check on %d mini-batches for at most %d epochs.",
            len(batches),
            self.overfit_max_epochs,
        )

        original_state = copy.deepcopy(self.model.state_dict())
        original_temperature = self.model.log_temperature.detach().clone()

        overfit_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        try:
            for epoch in range(1, self.overfit_max_epochs + 1):
                epoch_loss = 0.0
                for batch in batches:
                    self.model.train()
                    overfit_optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = self._forward_batch(batch, training=True, collect_metrics=False)
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    overfit_optimizer.step()
                    epoch_loss += float(loss.detach().item())

                epoch_loss /= max(len(batches), 1)
                logger.info(
                    "Overfit sanity epoch %d/%d | loss=%.6f",
                    epoch,
                    self.overfit_max_epochs,
                    epoch_loss,
                )
                if epoch_loss < 1e-3:
                    logger.info("Overfit sanity check converged; stopping early.")
                    break
        finally:
            self.model.load_state_dict(original_state)
            self.model.log_temperature.data.copy_(original_temperature)
            self.model.to(self.device)
            logger.info("Restored model weights after overfit sanity check.")

    def on_train_begin(self) -> None:  # type: ignore[override]
        logger.info(
            "Starting similarity transformer training for %d epochs | lr=%.2e | weight_decay=%.1e | grad_clip=%s | label_smoothing=%.3f",
            self.config.num_epochs,
            self.config.learning_rate,
            self.config.weight_decay,
            f"{self.gradient_clip_norm:.2f}" if self.gradient_clip_norm is not None else "disabled",
            self.label_smoothing,
        )
        logger.info(
            "Hard negatives: top_k=%d margin=%.3f weight=%.3f | temperature bounds=[%.3f, %.3f]",
            self.hard_negative_top_k,
            self.hard_negative_margin,
            self.hard_negative_weight,
            getattr(self.config, "temperature_min", 0.0),
            getattr(self.config, "temperature_max", 0.0),
        )

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
