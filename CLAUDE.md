# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: See `.claude/memory/project.md` for critical November 2025 updates on the `_forward_batch()` hook and combined similarity dataloader. Do NOT create artifact .md files - add information to memory instead.

## Project Overview

This is a research thesis project on pre-trained model recommendation for specific hardware using transformers. The system learns to recommend which pre-trained models will perform best on given datasets by:

1. Extracting feature vectors from model parameters using an autoencoder
2. Extracting dataset features using CLIP embeddings
3. Training a transformer model to predict model-dataset compatibility scores

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Training Commands

List all available trainers:
```bash
python train.py --list
```

Run specific trainers:
```bash
python train.py ModelAutoEncoderTrainer # Train the autoencoder for model embeddings
python train.py TransformerTrainer      # Train the ranking transformer
```

Trainer names are case-insensitive and the "Trainer" suffix is optional, so these work too:
```bash
python train.py modelautoencoder
python train.py transformer
```

### Evaluation and Extraction

Extract model parameter vectors:
```bash
python extract_model_vectors.py
```

Evaluate autoencoder embeddings:
```bash
python evaluate_model_autoencoder.py
```

Evaluate CLIP on ImageNet datasets:
```bash
python evaluate_clip_imagenet.py
```

### SLURM Cluster Execution

For cluster environments (Purdue Anvil):
```bash
sbatch train_autoencoder.sbatch                # Submit autoencoder training job
sbatch train_autoencoder.gautschi.sbatch       # Alternative configuration
```

## Architecture Overview

### Core Training System

The training system is built around a registry pattern with `TRAINER_REGISTRY` in `train/__init__.py`. All trainers inherit from `Trainer` in `train/base_trainer.py`, which provides:
- Training loop with progress tracking via tqdm
- Automatic metric logging and history persistence as JSON
- Configurable early stopping and learning rate scheduling
- Gradient clipping support
- Automatic plot generation for loss curves (linear, log, rolling mean)
- **New**: Multi-tensor input support via `_forward_batch()` hook for complex batch handling

**Trainer Implementation Patterns:**

1. **Simple tensor input** - Implement `compute_loss(batch: torch.Tensor)` (no override needed)
2. **Dict/dataclass input** - Implement `compute_loss(batch: dict)` or `compute_loss(batch: dataclass)` (no override needed)
3. **Complex batch handling** - Override `_forward_batch(batch)` for full control over multi-tensor inputs

The `_forward_batch()` method is called by the training loop and must return a scalar loss tensor. Default implementation calls `compute_loss()` for backward compatibility. Child trainers can override it to handle complex batch structures, multiple tensor inputs, or custom preprocessing logic.

**Backward Compatibility**: All existing trainers work without changes. The `_forward_batch()` hook is purely optional for new code requiring advanced batch handling.

### Configuration System

All configuration is centralized in `config.ini` and parsed through `config/parser.py`. The `ConfigParser` class:
- Loads `config.ini` from the project root
- Converts INI values to Python types (bool, int, float, lists, dicts)
- Provides section-specific config dataclasses via `ConfigParser.get(ConfigClass)`

Each subsystem (extractors, autoencoder, training, datasets) has its own section in `config.ini` and corresponding dataclass in `config/`.

### Model Components

**AutoEncoder** (`model/model_encoder.py`):
- Compresses 8192-dim model parameter vectors to 512-dim embeddings
- Configurable encoder/decoder architectures with hidden layers
- Supports GELU, ReLU, LeakyReLU activations with dropout
- `ModelAutoEncoder` subclass adds config-driven initialization

**RankingCrossAttentionTransformer** (`model/ranking_transformer.py`):
- Cross-attention transformer for model-dataset ranking
- Uses dataset_tokens as source (encoder input) and model_tokens as target (decoder input)
- Outputs logits for ranking the models

**CustomSimilarityTransformer** (`model/custom_similarity_transformer.py`):
- Cross-attention based model selector using PyTorch MultiheadAttention
- Learns to attend over model embeddings based on dataset features
- Produces probability distribution over models

**ClipImageEncoder** (`model/clip_encoder.py`):
- Extracts dataset features using OpenAI CLIP models
- Normalizes features and outputs fixed-dimensional embeddings

### Data Loading

**ModelParameterDataset** (`dataloader/model_npz_dataset.py`):
- Loads model parameter NPZ files for autoencoder training
- Flattens and normalizes parameter tensors

**Generic Dataset Loaders** (`dataloader/imagenet_dataset.py`):
- `GenericImageDataset` and `ImageNetDataset` for HuggingFace datasets
- `ClassBalancedBatchSampler` ensures balanced class distribution
- Registry-based configuration in `config.ini` under `[dataset_loader]`

**Similarity Datasets** (`dataloader/similarity_datasets.py`):
- `ModelEmbeddingDataset` loads pre-computed model embeddings
- `DatasetTokenDataset` loads CLIP-extracted dataset features in sharded NPZ format
- Both support batching and multiple splits (train/validation/test)

**Combined Similarity Dataloader** (`dataloader/combined_similarity_dataloader.py`) - **NEW**:
- `CombinedSimilarityDataset` - Unified loader for similarity transformer training
- `build_combined_similarity_loader()` - Factory function for DataLoader creation
- Returns batches with complete structure: `(dataset_tokens, model_tokens, true_ranks, model_names, dataset_names, model_indices, dataset_splits, batch_size)`
- Automatically loads ground truth rankings from `constants/dataset_model_performance.json`
- Handles variable-length dataset tokens with automatic padding
- Replaces need for manual coordination of multiple separate loaders

### Feature Extraction Pipeline

**Extractors** (`extractors/`):
- `HuggingFacePipelineExtractor`: Generic extractor using HF pipelines
- `SiglipExtractor`: Specialized for SigLIP models
- `DetectorBasedExtractor`: Uses detection-based models for parameter extraction
- Registry-driven selection via `config.ini` `[extractor]` section

### Loss Functions

**RankingLoss** (`loss/ranking_loss.py`):
- Compares model predictions against true performance rankings
- Used by `TransformerTrainer` to learn model-dataset compatibility

## Key File Locations

**Configuration**: `config.ini` - Runtime settings for all components
**Constants**: `constants/dataset_similarity.json`, `constants/dataset_model_performance.json` - Pre-computed metrics
**Artifacts**: `artifacts/` - Trained model weights, extracted features, training run logs
**Run Logs**: `artifacts/models/*/runs/*.json` - Training history with metrics

## Recent Improvements (November 2025)

### 1. Enhanced BaseTrainer for Multi-Tensor Inputs

**File Modified**: `train/base_trainer.py`

Added `_forward_batch()` hook method to support models with multiple tensor inputs without requiring changes to existing trainers:

```python
def _forward_batch(self, batch: Any) -> torch.Tensor:
    """Override this method for complex batch handling.

    Default implementation calls compute_loss() for backward compatibility.
    """
    return self.compute_loss(batch)
```

**Benefits**:
- Existing trainers work without modification
- New trainers can override `_forward_batch()` for complex batch structures
- No code duplication in training loop (`_train_batch()`, `_evaluate_batch()`)
- Supports dict batches, dataclass batches, and custom batch objects
- Full separation between loop infrastructure and batch-specific logic

**Documentation**: See `.claude/memory/project.md` for implementation patterns and examples

### 2. New Combined Similarity Dataloader

**File Created**: `dataloader/combined_similarity_dataloader.py`

Unified dataloader that replaces separate model and dataset loaders:

**Key Features**:
- Single coordinated loader instead of two separate ones
- Batch structure: `{"dataset_tokens": Tensor, "model_tokens": Tensor, "true_ranks": Tensor, ...}`
- Automatically loads ground truth rankings from `constants/dataset_model_performance.json`
- Handles variable-length dataset tokens with automatic padding
- Rich metadata (model names, dataset names, split tracking)
- Type-safe with TypedDict definitions

**Usage**:
```python
from dataloader import build_combined_similarity_loader

loader = build_combined_similarity_loader(splits=("train",), shuffle=True)
for batch in loader:
    probs = model(batch["model_tokens"], batch["dataset_tokens"])
    loss = ranking_loss(probs, batch["true_ranks"])
```

**Documentation**: See `.claude/memory/project.md` for usage patterns and batch structure reference

## Important Implementation Details

- All paths in `config.ini` are absolute (e.g., `/scratch/gautschi/patil185/`) and must be adjusted for local development
- Training history is automatically persisted to JSON after each epoch
- Model weights are saved with loss in filename: `autoencoder_weights.loss_0.008648.20251103_085137.pt`
- Dataset loaders use `beautilog` for logging; avoid print statements
- The autoencoder's code layer can have L1 regularization via `code_l1_penalty` config
- Reconstruction loss supports both MSE and Smooth L1 (configurable via `reconstruction_loss`)
- The `_forward_batch()` hook should return a scalar loss tensor; use `_collect_batch_metrics()` for auxiliary metrics
- The combined similarity dataloader reads rankings from `constants/dataset_model_performance.json`; ensure this file is populated before training

## Specialized Agents

This project includes specialized Claude Code agents for common tasks:

### trainer-expert
Generate production-ready PyTorch trainers that integrate with the BaseTrainer infrastructure. Use when creating new trainers, updating existing trainers to new patterns, or debugging trainer implementations.

**Trigger examples**:
- "Create a trainer for [ModelName]"
- "I need to train the ranking transformer"
- "Update my trainer to use the combined dataloader"

**Key capabilities**: Generates complete trainer classes with config dataclasses, ensures BaseTrainer patterns, validates model I/O contracts, recommends optimizers/schedulers.

### model-expert
Analyze and optimize PyTorch model architectures. Use when designing models, debugging training issues, or validating forward pass logic and gradient flow.

**Trigger examples**:
- "Review this transformer architecture"
- "My autoencoder loss is exploding"
- "Recommend hyperparameters for training"

**Key capabilities**: Reviews forward pass logic, analyzes gradient flow, estimates memory requirements, recommends loss functions.

### dataloader-expert
Analyze and optimize PyTorch dataloaders and data pipelines. Use when designing dataloaders, diagnosing I/O bottlenecks, or validating data quality.

**Trigger examples**:
- "Optimize my dataloader performance"
- "What batch structure does this dataset provide?"
- "Training is I/O bound, how can I improve throughput?"

**Key capabilities**: Analyzes dataset composition, recommends batch specifications, optimizes num_workers/pin_memory, validates data quality.
