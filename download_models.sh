#!/bin/bash
# Script to pre-download HuggingFace models using huggingface-cli

# Install huggingface_hub if not already installed
# pip install huggingface_hub

# Download models one by one
echo "Starting model downloads..."

# Your specific models
huggingface-cli download timm/test_efficientnet_ln.r160_in1k
huggingface-cli download timm/test_efficientnet_gn.r160_in1k
huggingface-cli download timm/test_vit2.r160_in1k
huggingface-cli download timm/test_resnet.r160_in1k
huggingface-cli download timm/test_vit.r160_in1k

echo "All models downloaded successfully!"
