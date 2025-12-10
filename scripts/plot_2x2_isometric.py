import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from seaborn import set_theme

set_theme(style="whitegrid")

# ==========================
# Data: 2x2 Matrix
# ==========================
# Matrix represents: Known/Unknown Models vs Known/Unknown Datasets
# Rows: Model Status (Known, Unknown/Blind)
# Cols: Dataset Status (Known, Unknown/Blind)
matrix = np.array([
    [0.86, 0.396],   # Known Models: [Known Datasets, Blind Datasets]
    [-0.029, 0.051]    # Blind Models: [Known Datasets, Blind Datasets]
])

# Labels for the axes
model_labels = ["Known\nModels", "Blind\nModels"]
dataset_labels = ["Known\nDatasets", "Blind\nDatasets"]

# Condition names for legend
condition_names = [
    "Known Models\nKnown Datasets",
    "Blind Models\nKnown Datasets",
    "Known Models\nBlind Datasets",
    "Blind Models\nBlind Datasets"
]

# ==========================
# MATERIAL Color Palette (from plot_results.py)
# ==========================
# Using (fill, stroke) tuples from plot_results.py palette
palette = [
    ("#D1EFEC", "#1AAE9F"),  # mint - Known Models, Known Datasets
    ("#F2D6F6", "#BD34D1"),  # blue - Known Models, Blind Datasets
    ("#FAE6D8", "#E8833A"),  # orange - Blind Models, Known Datasets
    ("#F6DADE", "#D3455B")   # red - Blind Models, Blind Datasets
]

# Extract fill and stroke colors
fill_colors = [p[0] for p in palette]
stroke_colors = [p[1] for p in palette]

# ==========================
# Setup 3D Plot
# ==========================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create bar positions
xpos, ypos = np.meshgrid(np.arange(2), np.arange(2))
xpos = xpos.flatten()
ypos = ypos.flatten()

# Bar dimensions
dx = dy = 0.6
dz = matrix.flatten()

# ==========================
# Create 3D Bars
# ==========================
for i in range(4):
    # Handle negative values by adjusting the starting z position
    if dz[i] < 0:
        z_start = dz[i]
        height = abs(dz[i])
        bar_alpha = 0.5  # More transparent for negative values
    else:
        z_start = 0
        height = dz[i]
        bar_alpha = 1.0  # Fully opaque for positive values

    ax.bar3d(
        xpos[i], ypos[i], z_start,
        dx, dy, height,
        color=fill_colors[i],
        edgecolor=stroke_colors[i],
        linewidth=2.0,
        alpha=bar_alpha,
        shade=False,
        lightsource=None
    )

# ==========================
# Isometric View Configuration
# ==========================
# Isometric view: elevation=35.264°, azimuth=45°
ax.view_init(elev=25, azim=45, roll=0)

# ==========================
# Axis Labels and Formatting
# ==========================
ax.set_xlabel('\nModel Status', fontsize=16, labelpad=10, weight='bold')
ax.set_ylabel('\nDataset Status', fontsize=16, labelpad=10, weight='bold')
ax.set_zlabel('Performance Score', fontsize=16, labelpad=10, weight='bold')

# Set tick positions and labels
ax.set_xticks([0.3, 1.3])
ax.set_xticklabels(dataset_labels, fontsize=13)
ax.set_yticks([0.3, 1.3])
ax.set_yticklabels(model_labels, fontsize=13)

# Z-axis formatting
ax.set_zlim(-0.2, 1.1)
ax.set_zticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# ==========================
# Title and Grid
# ==========================
ax.set_title('Model-Dataset Testing Conditions\nPerformance Matrix',
             fontsize=16, pad=20, weight='bold')

# Customize grid
ax.grid(True, alpha=0.7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

import mpl_toolkits.mplot3d.art3d as art3d
# Add a reference plane at z=0
from matplotlib.patches import Rectangle

rect = Rectangle((-0.5, -0.5), 3, 3, facecolor='lightgray', alpha=0.4, edgecolor='black', linewidth=0.5)
ax.add_patch(rect)
art3d.pathpatch_2d_to_3d(rect, z=0.0, zdir="z")

# ==========================
# Add value labels on top/bottom of bars
# ==========================
for i in range(4):
    # For negative values, place label at the bottom of the bar (below z=0)
    # For positive values, place label at the top of the bar (above z=0)
    if dz[i] < 0:
        # Negative bar: label goes below the bar
        label_z = dz[i] - 0.05
    else:
        # Positive bar: label goes above the bar
        label_z = dz[i] + 0.08

    ax.text(
        xpos[i] + dx/2,
        ypos[i] + dy/2,
        label_z,
        f'{dz[i]:.3f}',
        ha='center',
        va='center',
        fontsize=15,
        weight='bold',
        color='black',
        zorder=1000  # Ensure labels are drawn on top
    )

# ==========================
# Legend
# ==========================
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=fill_colors[0], edgecolor=stroke_colors[0], label=condition_names[0], fill=True),
    Patch(facecolor=fill_colors[1], edgecolor=stroke_colors[1], label=condition_names[1], fill=True),
    Patch(facecolor=fill_colors[2], edgecolor=stroke_colors[2], label=condition_names[2], fill=True),
    Patch(facecolor=fill_colors[3], edgecolor=stroke_colors[3], label=condition_names[3], fill=True)
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
          fontsize=13, framealpha=0.9)

plt.tight_layout()

# ==========================
# Save and Show
# ==========================
plt.savefig('/Users/parth/ptm-recommendation-with-transformers/scripts/2x2_matrix_isometric.png',
            dpi=300, bbox_inches='tight')
print("Plot saved to: scripts/2x2_matrix_isometric.png")
plt.show()
