# Project Memory: Pre-trained Model Recommendation with Transformers

## CRITICAL UPDATES (November 2025)

### 1. BaseTrainer _forward_batch() Hook (NEW)

**What Changed**:
- BaseTrainer now has a `_forward_batch(batch)` hook method for multi-tensor input support
- Default implementation: `_forward_batch()` calls `compute_loss()` (backward compatible)
- Can override: Trainers override `_forward_batch(batch)` for complex batch handling
- Called by: Both `_train_batch()` and `_evaluate_batch()` - eliminates loop duplication

**Key Benefits**:
- No need to override `_train_batch()` or `_evaluate_batch()` anymore
- Supports simple tensors, dicts, dataclasses, and custom batch objects
- No code duplication for complex batch handling

**When to Use Each**:
1. **Simple tensor input** → Implement `compute_loss(batch: torch.Tensor)`
2. **Dict/dataclass input** → Implement `compute_loss(batch: dict|dataclass)` - no override needed
3. **Complex multi-tensor** → Override `_forward_batch(batch)` - store metrics in `self._last_batch_metrics`

### 2. Combined Similarity Dataloader (NEW)

**What It Does**:
- Unified loader for similarity transformer training
- Factory function: `build_combined_similarity_loader(splits=("train",))` from `dataloader` module
- Replaces two separate loaders (ModelEmbeddingDataset + DatasetTokenDataset)

**Batch Structure** (dict with):
```python
{
    "dataset_tokens": Tensor,        # Shape: (B, M, D) - dataset embeddings
    "model_tokens": Tensor,          # Shape: (B, N, D) - model embeddings
    "true_ranks": Tensor,            # Shape: (B, N) - GROUND TRUTH RANKINGS (NEW!)
    "model_names": list[list[str]],  # Model names for tracking
    "dataset_names": list[str],      # Dataset names
    "model_indices": Tensor,         # (B, N) indices
    "dataset_splits": list[str],     # Split info
    "batch_size": int,               # Batch size
}
```

**Key Benefits**:
- Ground truth rankings automatically loaded from `constants/dataset_model_performance.json`
- Guaranteed alignment (no manual coordination of two loaders)
- Automatic padding of variable-length dataset tokens
- Cleaner trainer code

**When to Use**:
- For any similarity/ranking trainer
- Recommended over manual coordination of separate loaders

### 3. Implementation Patterns Update

Trainers should now follow one of these patterns:

**Pattern 1 (Simple Tensor)**:
```python
def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
    output = self.model(batch)
    return F.mse_loss(output, targets)
```

**Pattern 2 (Dict/Dataclass)**:
```python
def compute_loss(self, batch: dict) -> torch.Tensor:
    data = batch["data"].to(self.device)
    targets = batch["targets"].to(self.device)
    output = self.model(data)
    return F.mse_loss(output, targets)
```

**Pattern 3 (Complex Multi-Tensor)** - Override _forward_batch():
```python
def _forward_batch(self, batch: MyBatch) -> torch.Tensor:
    batch = batch.to(self.device)
    output = self.model(batch.data1, batch.data2, batch.data3)
    loss = F.some_loss(output, batch.targets)
    self._last_batch_metrics = {"metric1": value1}
    return loss

def _collect_batch_metrics(self) -> dict:
    return dict(self._last_batch_metrics)
```

**Pattern 4 (Similarity with Combined Loader)**:
```python
from dataloader import build_combined_similarity_loader

loader = build_combined_similarity_loader(splits=("train",))
# Batch includes: dataset_tokens, model_tokens, true_ranks, etc.

def compute_loss(self, batch: dict) -> torch.Tensor:
    probs = self.model(batch["model_tokens"], batch["dataset_tokens"])
    return ranking_loss(probs, batch["true_ranks"])
```

## Architecture Reminders

### BaseTrainer Features
- Training loop with tqdm progress tracking
- Automatic JSON metric logging and history persistence
- Configurable early stopping and learning rate scheduling
- Gradient clipping support
- Automatic plot generation (linear, log, rolling mean)
- `_forward_batch()` hook for custom batch handling (NEW)

### Configuration System
- Centralized in `config.ini`
- All paths are absolute (must adjust for local development)
- ConfigParser in `config/parser.py` parses to Python types
- Each subsystem has section in config.ini and dataclass in `config/`

### Models
- **AutoEncoder** (`model/model_encoder.py`): 8192→512 dim compression
- **RankingCrossAttentionTransformer** (`model/ranking_transformer.py`): Cross-attention transformer for model-dataset ranking
- **CustomSimilarityTransformer** (`model/custom_similarity_transformer.py`): Cross-attention based model selector
- **ClipImageEncoder** (`model/clip_encoder.py`): Dataset feature extraction

### Data Loading
- **ModelParameterDataset**: NPZ files for autoencoder training
- **ModelEmbeddingDataset**: Pre-computed model embeddings
- **DatasetTokenDataset**: CLIP features in sharded NPZ
- **CombinedSimilarityDataset** (NEW): Unified loader with ground truth

### Loss Functions
- **RankingLoss** (`loss/ranking_loss.py`): For model-dataset compatibility

### Logging
- Use `beautilog.logger` - never print()
- Project uses beautilog for all logging

### Paths
- Always use `pathlib.Path`
- All config paths are absolute
- Never use string paths

## Backward Compatibility

✅ **All existing code continues to work**:
- Existing trainers work without changes
- _forward_batch() is optional override
- Combined dataloader is optional
- Old separate loaders still work

❌ **No breaking changes made**

## Performance Notes

- AutoEncoder training uses MSE or Smooth L1 (configurable via `reconstruction_loss`)
- AutoEncoder code layer can have L1 regularization via `code_l1_penalty`
- Model weights saved with format: `autoencoder_weights.loss_0.008648.20251103_085137.pt`
- Training history persisted to JSON after each epoch
- Metrics automatically logged and plotted

## Specialized Agents

### trainer-expert
- **Purpose**: Generates production-ready PyTorch trainers for this project
- **When to use**: Creating new trainers, debugging existing ones, updating trainers to new patterns
- **Key capabilities**:
  - Generates complete trainer classes with config dataclasses
  - Ensures BaseTrainer integration patterns are followed
  - Recommends appropriate optimizers/schedulers/loss functions
  - Validates model I/O contracts before generation
- **Trigger examples**: "Create a trainer for X", "Update my trainer to use combined dataloader"

### model-expert
- **Purpose**: Analyzes PyTorch neural network architectures
- **When to use**: Designing/optimizing models, debugging training issues, validating architectures
- **Key capabilities**: Reviews forward pass logic, analyzes gradient flow, recommends hyperparameters
- **Trigger examples**: "Review this transformer", "My loss is exploding"

### dataloader-expert
- **Purpose**: Analyzes and optimizes PyTorch dataloaders and data pipelines
- **When to use**: Designing dataloaders, diagnosing I/O bottlenecks, validating data quality
- **Key capabilities**: Analyzes dataset composition, recommends batch specs, optimizes throughput
- **Trigger examples**: "Optimize dataloader performance", "Training is I/O bound"

## Thesis Writing (December 2025)

### Section 1.2 Motivation - Recent Refinement

**Status**: Refined with professor feedback, ready for figure exports

**Key Files**:
- `thesis/section-1-2-refined.tex` - Complete refined motivation section (~650 words)
- `thesis/SECTION_1-2_SETUP_GUIDE.md` - Comprehensive setup and integration guide
- `thesis/figures/` - Directory created for PowerPoint figure exports (lowercase to match LaTeX paths)
- `thesis/references-motivation.bib` - 9 citations supporting motivation section
- `thesis.pptx` - Defense presentation (source for figures on slides 5-6, 8)

**What's Included**:
- Two figure placeholders with proper captions (waiting for PowerPoint exports)
- Formal "Problem Formulation" subsection with mathematical notation
- Input: Model zoo M, dataset D, hardware constraints H
- Output: Ranked list π with performance inequality
- Four explicit assumptions
- SOTA limitations: Model Spider (learnable tokens), EMMS (metadata-based), PTMPicker (no hardware awareness)
- 11 citations (9 verified, 2 missing - see below)

**Verified Citations** (in .bib files):
- `wolf2020transformers` - HuggingFace Transformers library
- `hfcommunity2024dataset` - HFCommunity dataset on Zenodo
- `jiang2024peatmoss` - PeaTMOSS: Mining PTMs in OSS
- `ding2024ptmtorrent` - PTMTorrent cross-hub aggregation
- `schwartz2020green` - Green AI
- `wu2022sustainable` - Sustainable AI
- `zhang2023modelspider` - Model Spider (NeurIPS 2023 baseline)
- `meng2023foundation` - EMMS foundation model selector
- `liu2025ptmpicker` - PTMPicker tool

**Missing Citations** (need to add or remove):
- `jajal2024interoperability` - Line 13 (model repository scale)
- `jiang_empirical_2023` - Line 64 (large-scale PTM usage)

**Action Items Remaining**:
1. Export Figure 1: `thesis/figures/ptm_recommendation_problem_overview.png` from slides 5-6
2. Export Figure 2: `thesis/figures/huggingface_kaggle_growth.png` from slide 8
3. Handle missing citations (add to .bib or remove from .tex)
4. Compile and verify

### Acknowledgments Section

**Status**: Refined December 2025

**File**: `thesis/acknowledgments-refined.tex`

**Key Features**:
- Informal but polished tone
- Personal acknowledgments (self, family, advisor, collaborators)
- Memorable touches ("Pika pika!" for Huiyun Peng)
- Institutional support line included

**Advisor**: Professor James Davis (verify name if incorrect)

### Thesis Structure Notes

- All LaTeX includes use lowercase `figures/` paths
- Bibliography files are modular by section (references-motivation.bib, references-2-1.bib, etc.)
- Chapter labels referenced in Section 1.2: `\label{sec:motivation}`, `\label{subsec:problem-formulation}`
- System name: Cross-Sight / Cross-Select (working title)

## Important Files

- **CLAUDE.md**: Main project documentation (updated with new sections)
- **train/base_trainer.py**: Has _forward_batch() hook
- **dataloader/combined_similarity_dataloader.py**: Combined loader implementation
- **config.ini**: All configuration
- **.claude/agents/trainer-expert.md**: Generates production-ready trainers
- **.claude/agents/model-expert.md**: Analyzes architectures
- **.claude/agents/dataloader-expert.md**: Optimizes data pipelines
- **thesis/section-1-2-refined.tex**: Refined motivation section (Dec 2025)
- **thesis/SECTION_1-2_SETUP_GUIDE.md**: Complete integration guide
- **thesis/acknowledgments-refined.tex**: Refined acknowledgments (Dec 2025)

## DO NOT

- Create artifact .md files - add to memory instead
- Use print() - use beautilog.logger
- Use string paths - use pathlib.Path
- Duplicate _train_batch()/_evaluate_batch() - override _forward_batch() if needed
- Manually coordinate multiple loaders - use combined dataloader for similarity tasks
- Re-implement training loop infrastructure - use BaseTrainer hooks
- Reference external documentation files - point to .claude/memory/project.md instead
