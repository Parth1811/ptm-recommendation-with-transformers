# Layer Clustering Visualization

This animation script visualizes the clustering process used in the model parameter extraction pipeline.

## Overview

The script demonstrates how raw model layer parameters are compressed using K-means clustering, which is a key step in creating fixed-size model embeddings.

## Scenes

### 1. LayerClusteringVisualization (Main Scene)
Shows a side-by-side comparison of:
- **Left panel**: Histogram of raw parameter values from a model layer
- **Right panel**: Histogram of cluster centers after K-means clustering
- **Arrow**: Shows the compression process with cluster count

**Features**:
- Loads a Vision Transformer (ViT) model from HuggingFace
- Extracts a weight layer (first attention layer by default)
- Samples 1000 values for visualization
- Applies K-means clustering to compress to 64 cluster centers
- Displays statistics (mean, std, min, max) for both distributions

### 2. LayerClusteringDetailed
A step-by-step visualization showing:
1. Raw parameter values as scatter plot
2. K-means clustering application
3. Resulting cluster centers
4. Compression ratio visualization

## Usage

Render the main visualization:
```bash
cd animations
manim -pql layer_clustering_viz.py LayerClusteringVisualization
```

Render the detailed step-by-step version:
```bash
manim -pql layer_clustering_viz.py LayerClusteringDetailed
```

High quality render:
```bash
manim -pqh layer_clustering_viz.py LayerClusteringVisualization
```

## Configuration

You can modify these parameters in the `construct()` method:

```python
model_id = "google/vit-base-patch16-224"  # HuggingFace model ID
n_clusters = 64                           # Number of cluster centers
layer_name = "encoder.layer.0.attention.attention.query.weight"  # Layer to visualize
sample_size = 1000                        # Number of values to sample
```

## Technical Details

### Clustering Process
The script uses sklearn's KMeans algorithm (instead of FAISS GPU version) for better compatibility:
1. Samples random values from the flattened layer tensor
2. Applies K-means clustering with k=64 (configurable)
3. Sorts cluster centers in descending order
4. Visualizes distributions as histograms

### Histogram Visualization
- 30 bins for both raw and clustered distributions
- Normalized bar heights for easy comparison
- Material design color scheme matching project style
- Statistics panel below each histogram

## Dependencies

- manim (for animation)
- transformers (for loading models)
- torch (for model operations)
- sklearn (for K-means clustering)
- numpy (for array operations)

## Related Files

- `extractors/base_extractor.py` - Contains the production clustering code
- `extractors/hf_pipeline_extractor.py` - HuggingFace model loading
- `model_pipeline.py` - Shows the full model processing pipeline
- `config.ini` - Configuration for the extraction system
