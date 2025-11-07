# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python train.py ModelAutoEncoderTrainer      # Train the autoencoder for model embeddings
python train.py SimilarityTransformerTrainer # Train the similarity transformer
```

Trainer names are case-insensitive and the "Trainer" suffix is optional, so these work too:
```bash
python train.py modelautoencoder
python train.py similarity
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

Subclass trainers must implement `compute_loss(batch)` and can override hooks like `on_train_begin()` and `_collect_batch_metrics()`.

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

**SimilarityTransformerModel** (`model/similarity_transformer.py`):
- Takes model embeddings (fixed set) and dataset token embeddings (variable length batches)
- Projects both to common hidden dimension
- Uses BERT-based transformer with CLS token for classification
- Outputs logits ranking which models are best for the dataset

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

### Feature Extraction Pipeline

**Extractors** (`extractors/`):
- `HuggingFacePipelineExtractor`: Generic extractor using HF pipelines
- `SiglipExtractor`: Specialized for SigLIP models
- `DetectorBasedExtractor`: Uses detection-based models for parameter extraction
- Registry-driven selection via `config.ini` `[extractor]` section

### Loss Functions

**RankingLoss** (`loss/ranking_loss.py`):
- Compares model predictions against true performance rankings
- Used by `SimilarityTransformerTrainer` to learn model-dataset compatibility

## Key File Locations

**Configuration**: `config.ini` - Runtime settings for all components
**Constants**: `constants/dataset_similarity.json`, `constants/dataset_model_performance.json` - Pre-computed metrics
**Artifacts**: `artifacts/` - Trained model weights, extracted features, training run logs
**Run Logs**: `artifacts/models/*/runs/*.json` - Training history with metrics

## Important Implementation Details

- All paths in `config.ini` are absolute (e.g., `/scratch/gautschi/patil185/`) and must be adjusted for local development
- Training history is automatically persisted to JSON after each epoch
- Model weights are saved with loss in filename: `autoencoder_weights.loss_0.008648.20251103_085137.pt`
- Dataset loaders use `beautilog` for logging; avoid print statements
- The autoencoder's code layer can have L1 regularization via `code_l1_penalty` config
- Reconstruction loss supports both MSE and Smooth L1 (configurable via `reconstruction_loss`)
