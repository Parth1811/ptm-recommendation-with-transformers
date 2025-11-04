"""Training harness for the similarity transformer model."""

from __future__ import annotations

from typing import Callable

import torch
from beautilog import logger

from config import (ConfigParser, DatasetTokenLoaderConfig,
                    ModelEmbeddingLoaderConfig, SimilarityModelConfig,
                    SimilarityTrainerConfig)
from dataloader import (DatasetTokenDataset, ModelEmbeddingDataset,
                        build_dataset_token_loader,
                        build_model_embedding_loader)
from model import SimilarityTransformerModel


class SimilarityTransformerTrainer:
    """Skeleton trainer; plug in a custom loss function before training."""

    def __init__(self) -> None:

        self.trainer_cfg = ConfigParser.get(SimilarityTrainerConfig)
        self.model_cfg = ConfigParser.get(SimilarityModelConfig)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimilarityTransformerModel(self.model_cfg)
        self.model.to(self.device)

        self.loss_fn = None  # Placeholder for the loss function

        self.model_loader = build_model_embedding_loader()
        self.model_tokens = self._gather_model_tokens().to(self.device).detach()

        self.dataset_loader = build_dataset_token_loader()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_cfg.learning_rate,
            weight_decay=self.trainer_cfg.weight_decay,
        )

    def compute_loss(self, batch: dict[str, object]) -> torch.Tensor:
        if self.loss_fn is None:
            raise NotImplementedError("Provide a loss_fn or override compute_loss for custom training logic.")
        return self.loss_fn(self.model, batch)

    def train(self) -> None:
        self.model.train()

        for epoch in range(int(self.trainer_cfg.num_epochs)):
            logger.info("Starting epoch %d", epoch + 1)

            for step, dataset_batch in enumerate(self.dataset_loader, start=1):
                prepared_batch = self._prepare_batch(dataset_batch)
                loss = self.compute_loss(prepared_batch)

                if not torch.is_tensor(loss):
                    raise TypeError("Loss function must return a torch.Tensor.")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.trainer_cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_cfg.grad_clip)

                self.optimizer.step()

                if step % max(1, self.trainer_cfg.log_every_n_epochs) == 0:
                    logger.info("Epoch %d Step %d Loss %.6f", epoch + 1, step, float(loss.item()))

    def _prepare_batch(self, dataset_batch: dict[str, object]) -> dict[str, object]:
        dataset_tokens = dataset_batch["dataset_tokens"]
        if not isinstance(dataset_tokens, torch.Tensor):
            raise TypeError("Dataset tokens must be a torch.Tensor")

        if dataset_tokens.dim() == 2:
            dataset_tokens = dataset_tokens.unsqueeze(0)
        elif dataset_tokens.dim() != 3:
            raise ValueError("Dataset tokens must have shape [batch, num_classes, embedding_dim]")

        batch_size = dataset_tokens.size(0)
        model_tokens = self.model_tokens.unsqueeze(0).repeat(batch_size, 1, 1)

        return {
            "dataset_tokens": dataset_tokens.to(self.device),
            "model_tokens": model_tokens.to(self.device),
        }

    def _gather_model_tokens(self) -> torch.Tensor:
        all_tokens: list[torch.Tensor] = []
        for batch in self.model_loader:
            model_tokens = batch["model_tokens"]
            if not isinstance(model_tokens, torch.Tensor):
                raise TypeError("Model loader must yield torch.Tensor under 'model_tokens'")
            all_tokens.append(model_tokens)

        if not all_tokens:
            raise ValueError("Model embedding loader did not yield any tokens.")

        return torch.cat(all_tokens, dim=0)
