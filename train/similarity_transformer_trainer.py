"""Training harness for the similarity transformer model."""

from __future__ import annotations

from itertools import cycle
from typing import Callable, Iterator

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

    def __init__(
        self,
        *,
        model: SimilarityTransformerModel | None = None,
        model_loader_cfg: ModelEmbeddingLoaderConfig | None = None,
        dataset_loader_cfg: DatasetTokenLoaderConfig | None = None,
        trainer_cfg: SimilarityTrainerConfig | None = None,
        loss_fn: Callable[[SimilarityTransformerModel, dict[str, object]], torch.Tensor] | None = None,
    ) -> None:
        ConfigParser.load()
        self.trainer_cfg = trainer_cfg or ConfigParser.get(SimilarityTrainerConfig)
        self.model_cfg = ConfigParser.get(SimilarityModelConfig)
        self.model_loader_cfg = model_loader_cfg or ConfigParser.get(ModelEmbeddingLoaderConfig)
        self.dataset_loader_cfg = dataset_loader_cfg or ConfigParser.get(DatasetTokenLoaderConfig)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model or SimilarityTransformerModel(self.model_cfg)
        self.model.to(self.device)

        self.loss_fn = loss_fn

        self.model_dataset = ModelEmbeddingDataset(
            self.model_loader_cfg.root_dir,
            embedding_key=self.model_loader_cfg.embedding_key,
            max_models=self.model_loader_cfg.max_models,
        )
        self.dataset_dataset = DatasetTokenDataset(
            self.dataset_loader_cfg.root_dir,
            dataset_names=self.dataset_loader_cfg.dataset_names,
            splits=self.dataset_loader_cfg.splits,
            shard_glob=self.dataset_loader_cfg.shard_glob,
            average_over_batches=self.dataset_loader_cfg.average_over_batches,
            include_class_metadata=self.dataset_loader_cfg.include_class_metadata,
        )

        self.model_loader = build_model_embedding_loader(self.model_dataset, config=self.model_loader_cfg)
        self.dataset_loader = build_dataset_token_loader(self.dataset_dataset, config=self.dataset_loader_cfg)

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
        model_iter: Iterator[dict[str, torch.Tensor | list[str]]] = cycle(self.model_loader)

        for epoch in range(int(self.trainer_cfg.num_epochs)):
            logger.info("Starting epoch %d", epoch + 1)
            for step, dataset_batch in enumerate(self.dataset_loader, start=1):
                model_batch = next(model_iter)

                prepared_batch = self._prepare_batch(dataset_batch, model_batch)
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

    def _prepare_batch(
        self,
        dataset_batch: dict[str, object],
        model_batch: dict[str, torch.Tensor | list[str]],
    ) -> dict[str, object]:
        tokens = dataset_batch["tokens"]
        if isinstance(tokens, torch.Tensor):
            tokens_tensor = tokens.unsqueeze(0) if tokens.dim() == 2 else tokens
        else:
            raise TypeError("Dataset tokens must be a torch.Tensor")

        model_embeddings = model_batch["embeddings"]
        if not isinstance(model_embeddings, torch.Tensor):
            raise TypeError("Model embeddings must be a torch.Tensor")

        batch = {
            "dataset_tokens": tokens_tensor.to(self.device),
            "model_embeddings": model_embeddings.to(self.device),
            "dataset_metadata": dataset_batch,
            "model_metadata": model_batch,
        }
        return batch
