import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_theme

set_theme(style="whitegrid")

# ==========================
# Data: Fine-tuning Time Required
# ==========================
# Time in minutes for fine-tuning on 1x A100 GPU
providers = [
    "Hugging Face\nPEFT",
    "Microsoft\nDeepSpeed",
    "Hugging Face\nTransformers",
    "Unsloth.ai",
]

# Fine-tuning time in minutes (example data for Llama-3.2-11B-Vision-Instruct, 300 steps)
finetuning_times = [
    190,  # Hugging Face PEFT
    175,  # Microsoft DeepSpeed
    175,  # Hugging Face Transformers
    82,   # Unsloth.ai
]

# ==========================
# MATERIAL Color Palette (from plot_results.py)
# ==========================
# Using (fill, stroke) tuples from plot_results.py palette
palette = [
    ("#D5E7F7", "#2C88D9"),   # blue - Hugging Face
    ("#FAE6D8", "#E8833A"),   # orange - Microsoft
    ("#D1EFEC", "#1AAE9F"),   # mint - Unsloth (fastest)
    ("#F6DADE", "#D3455B"),   # red - AWS
    ("#F2D6F6", "#BD34D1"),   # purple - Google Cloud
]

# Extract fill and stroke colors
fill_colors = [p[0] for p in palette]
stroke_colors = [p[1] for p in palette]

# ==========================
# Plotting
# ==========================
fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(providers))
bars = ax.bar(
    x,
    finetuning_times,
    width=0.65,
    color=fill_colors,
    edgecolor=stroke_colors,
    linewidth=2.0
)

# Add value labels on top of bars
for i, (bar, time) in enumerate(zip(bars, finetuning_times)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 3,
        f'{time:.1f} min',
        ha='center',
        va='bottom',
        fontsize=18,
        weight='bold'
    )

# ==========================
# Aesthetics
# ==========================
ax.set_xticks(x)
ax.set_xticklabels(providers, fontsize=16, weight='bold')
ax.set_ylabel('Fine-tuning Time (minutes)', fontsize=18, weight='bold')
ax.set_title('Fine-tuning Time Comparison Across Providers\nLlama-3.2-11B-Vision-Instruct | 1× NVIDIA A100 (80GB) | 300 steps',
             fontsize=18, pad=20, weight='bold')

# Set y-axis limits
ax.set_ylim(0, max(finetuning_times) * 1.15)

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Increase y-axis tick label size
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()

# Save
plt.savefig('/Users/parth/ptm-recommendation-with-transformers/scripts/finetuning_time_comparison.png',
            dpi=300, bbox_inches='tight')
print("Plot saved to: scripts/finetuning_time_comparison.png")
plt.show()
