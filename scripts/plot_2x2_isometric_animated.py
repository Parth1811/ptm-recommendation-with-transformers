import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
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
    ("#F2D6F6", "#BD34D1"),  # purple - Blind Models, Known Datasets
    ("#FAE6D8", "#E8833A"),  # orange - Known Models, Blind Datasets
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
dz_final = matrix.flatten()

# ==========================
# Animation Setup
# ==========================
num_frames = 120
frames = np.linspace(0, 1, num_frames)

# Easing function for smooth animation (ease-out cubic)
def ease_out_cubic(t):
    return 1 - np.power(1 - t, 3)

# Store bar objects and text objects
bar_patches = []
text_objects = []

def init():
    """Initialize the plot"""
    ax.clear()

    # Axis Labels and Formatting
    ax.set_xlabel('\nModel Status', fontsize=14, labelpad=10, weight='bold')
    ax.set_ylabel('\nDataset Status', fontsize=14, labelpad=10, weight='bold')
    ax.set_zlabel('Performance Score', fontsize=14, labelpad=10, weight='bold')

    # Set tick positions and labels
    ax.set_xticks([0.3, 1.3])
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.set_yticks([0.3, 1.3])
    ax.set_yticklabels(model_labels, fontsize=11)

    # Z-axis formatting
    ax.set_zlim(-0.2, 1.1)
    ax.set_zticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Title and Grid
    ax.set_title('Model-Dataset Testing Conditions\nPerformance Matrix',
                 fontsize=16, pad=20, weight='bold')

    # Customize grid
    ax.grid(True, alpha=0.7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Isometric view
    ax.view_init(elev=25, azim=45, roll=0)

    # Add reference plane at z=0
    import mpl_toolkits.mplot3d.art3d as art3d
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, -0.5), 3, 3, facecolor='lightgray', alpha=0.4,
                     edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0.0, zdir="z")

    return []

def update(frame):
    """Update function for animation"""
    # Clear previous bars and text
    ax.clear()

    # Re-initialize plot settings
    ax.set_xlabel('\nModel Status', fontsize=14, labelpad=10, weight='bold')
    ax.set_ylabel('\nDataset Status', fontsize=14, labelpad=10, weight='bold')
    ax.set_zlabel('Performance Score', fontsize=14, labelpad=10, weight='bold')

    ax.set_xticks([0.3, 1.3])
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.set_yticks([0.3, 1.3])
    ax.set_yticklabels(model_labels, fontsize=11)

    ax.set_zlim(-0.2, 1.1)
    ax.set_zticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_title('Model-Dataset Testing Conditions\nPerformance Matrix',
                 fontsize=16, pad=20, weight='bold')

    ax.grid(True, alpha=0.7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.view_init(elev=25, azim=45, roll=0)

    # Add reference plane
    import mpl_toolkits.mplot3d.art3d as art3d
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, -0.5), 3, 3, facecolor='black', alpha=0.4,
                     edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0.0, zdir="z")

    # Calculate progress with easing
    progress = ease_out_cubic(frame)

    # Draw bars with animated heights
    for i in range(4):
        # Current height for this frame
        current_dz = dz_final[i] * progress

        # Handle negative values
        if dz_final[i] < 0:
            z_start = current_dz
            height = abs(current_dz)
            bar_alpha = 0.5
        else:
            z_start = 0
            height = current_dz
            bar_alpha = 1.0

        # Draw bar
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

        # Add value labels (only show when bar is at least 50% grown)
        if progress > 0.5:
            if current_dz < 0:
                label_z = current_dz - 0.05
            else:
                label_z = current_dz + 0.08

            # Fade in the label
            label_alpha = min(1.0, (progress - 0.5) * 2)

            ax.text(
                xpos[i] + dx/2,
                ypos[i] + dy/2,
                label_z,
                f'{dz_final[i]:.3f}',
                ha='center',
                va='center',
                fontsize=13,
                weight='bold',
                color='black',
                alpha=label_alpha,
                zorder=1000
            )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=fill_colors[0], edgecolor=stroke_colors[0],
              label=condition_names[0], fill=True),
        Patch(facecolor=fill_colors[1], edgecolor=stroke_colors[1],
              label=condition_names[1], fill=True),
        Patch(facecolor=fill_colors[2], edgecolor=stroke_colors[2],
              label=condition_names[2], fill=True),
        Patch(facecolor=fill_colors[3], edgecolor=stroke_colors[3],
              label=condition_names[3], fill=True)
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.05, 1), fontsize=10, framealpha=0.9)

    return []

# ==========================
# Create Animation
# ==========================
print("Creating animation...")
anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                     blit=False, interval=20, repeat=False)

# ==========================
# Save Animation
# ==========================
print("Saving animation as GIF...")
writer = PillowWriter(fps=60)
anim.save('/Users/parth/ptm-recommendation-with-transformers/scripts/2x2_matrix_isometric_animated.gif',
          writer=writer, dpi=100)

print("Animation saved to: scripts/2x2_matrix_isometric_animated.gif")
print("You can also save as MP4 by uncommenting the lines below and commenting out the GIF save")

# To save as MP4 instead (requires ffmpeg):
from matplotlib.animation import FFMpegWriter

writer = FFMpegWriter(fps=60, bitrate=1800)
anim.save('/Users/parth/ptm-recommendation-with-transformers/scripts/2x2_matrix_isometric_animated.mp4',
          writer=writer, dpi=150)

plt.close()
print("Done!")
