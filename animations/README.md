# Manim Animations for ML Pipeline

This folder contains Manim animations for visualizing the recommendation pipeline components.

## Components

### 1. `round_box.py`
Utility class for creating rounded rectangle containers.

**Usage:**
```python
from animations.round_box import RoundBox
box = RoundBox("My Text", width=2, height=1, fill_color=BLUE)
```

### 2. `tokens.py`
Token visualization classes for model and dataset embeddings.

**Classes:**
- `Tokens` - Generic token visualization
- `ModelTokens` - Specialized for model tokens (512 x 1)
- `DatasetTokens` - Specialized for dataset tokens (512 x 1)

**Usage:**
```python
from animations.tokens import ModelTokens, DatasetTokens
model_tokens = ModelTokens(num_models=3)
dataset_tokens = DatasetTokens(num_samples=4)
```

### 3. `neural_network.py`
Neural network visualization with configurable layers and smooth forward pass animation.

**Features:**
- Configurable layer sizes
- Smooth color flow from nodes through edges to next layer
- Customizable flow color and timing
- Automatic centering

**Usage:**
```python
from neural_network import NeuralNetwork
nn = NeuralNetwork(layers=[4, 6, 6, 3])  # [input, hidden1, hidden2, output]
nn.animate_forward_pass(scene, run_time=4, flow_color=YELLOW)
```

### 4. `transformer.py`
Transformer and cross-attention block visualizations.

**Classes:**
- `CrossAttentionBlock` - Q, K, V attention mechanism
- `Transformer` - Full transformer with FC layer

**Usage:**
```python
from animations.transformer import Transformer, CrossAttentionBlock
attention = CrossAttentionBlock()
transformer = Transformer(show_fc_layer=True)
```

### 5. `dataset_pipeline.py`
Complete dataset processing pipeline.

**Usage:**
```python
from animations.dataset_pipeline import DatasetPipeline
pipeline = DatasetPipeline()
```

## Running Animations

### Run the main showcase (all components):
```bash
manim -pql main.py ShowcaseAll
```

### Run individual scenes:
```bash
# Neural Network only
manim -pql main.py NeuralNetworkScene

# Transformer only
manim -pql main.py TransformerScene

# Dataset Pipeline only
manim -pql main.py DatasetPipelineScene

# Tokens only
manim -pql main.py TokensScene
```

### Quality Options:
- `-ql` - Low quality (480p, fast)
- `-qm` - Medium quality (720p)
- `-qh` - High quality (1080p)
- `-qk` - 4K quality (2160p)

### Other Flags:
- `-p` - Preview after rendering
- `-s` - Save last frame as image
- `--format=gif` - Render as GIF

## Example: Custom Scene

```python
from manim import *
from animations.neural_network import NeuralNetwork
from animations.tokens import ModelTokens

class MyCustomScene(Scene):
    def construct(self):
        nn = NeuralNetwork(layers=[3, 5, 2])
        tokens = ModelTokens(num_models=2)
        tokens.next_to(nn, RIGHT, buff=1)

        self.play(Create(nn))
        self.play(Create(tokens))
        nn.animate_forward_pass(self, run_time=2)
        self.wait(2)
```

Then run:
```bash
manim -pql main.py MyCustomScene
```

## File Structure

```
animations/
├── round_box.py          # Utility: rounded rectangles
├── tokens.py             # Token visualizations
├── neural_network.py     # Neural network with layers
├── transformer.py        # Transformer & attention blocks
├── dataset_pipeline.py   # Dataset processing pipeline
├── main.py              # Showcase scenes
├── manim.cfg            # Manim configuration
└── README.md            # This file
```

## Tips

1. Use `VGroup()` to group multiple mobjects
2. Use `.scale()`, `.shift()`, `.next_to()` for positioning
3. All classes inherit from Manim's `VGroup` for easy composition
4. Customize colors using Manim constants: `BLUE`, `RED`, `GREEN`, `TEAL`, etc.
5. Adjust `fill_opacity` and `stroke_width` for visual effects
