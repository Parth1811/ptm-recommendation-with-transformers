import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import set_theme

set_theme(style="whitegrid")

# ==========================
# Data
# ==========================
# metrics = ["H-Score","NCE","LEEP","N-LEEP","LogME","PACTran","OTCE","LFC","ModelSpider","Cross-sight"]
# data = {
#     "Mean": [0.16, 0.426, 0.304, 0.429, 0.574, 0.385, 0.123, -0.064, 0.75, 0.86],
#     # "Cifar10": [0.018, 0.267, 0.53, 0.515, 0.718, 0.631, 0.515, 0.33, 0.87, 0.994],
#     # "Cifar100":[0.005, 0.232, 0.471, 0.707, 0.698, 0.614, 0.505, 0.271, 0.977, 0.705],
#     # "Caltech 101":[0.078, 0.534, -0.038, 0.476, 0.453, 0.262, -0.05, -0.226, 0.711, 0.713]
# }

# metrics = ["Known Models", "Blind Models"]
# data = {
#     "Cifar10": [0.994, 0.022],
#     "Cifar100": [0.705, 0.115],
#     "Caltech 101": [0.713, -0.016],
#     "FashionMNIST": [0.991, -0.033],
#     "FashionProduct": [0.993, -0.091],
#     "Deepfashion": [0.988, -0.148],
#     "Ham10000": [0.986, -0.102],
#     "SVHN": [0.982, -0.028],
#     "MNIST": [0.928, 0.007],
#     "USPS Digits": [0.915, 0.033],
#     "Imagenet 1k": [0.268, -0.076],
#     "Mean": [0.860, -0.029],
# }

metrics = ["Model Spider\n(Both Known)", "Blind Datasets", "Double Blind"]
data = {
    "Aircraft": [0.382, 0.085, -0.006],
    "Cars": [0.727, 0.232, 0.140],
    "Pets": [0.717, 0.006, -0.033],
    "DTD": [0.686, 0.923, 0.059],
    "SUN397": [0.933, 0.734, 0.096],
    "Mean": [0.689, 0.396, 0.051],
}



df = pd.DataFrame(data, index=metrics)

# ==========================
# MATERIAL Color Palette
# (muted fill, stroke border)
# ==========================
palette = [
    # ("#D5E7F7", "#2C88D9"),   # blue
    # ("#F6DADE", "#D3455B"),   # red
    # ("#FDF3D3", "#F7C325"),   # yellow
    # ("#E1E5EC", "#9EADBA"),   # dark gray
    # ("#FAE6D8", "#E8833A"),   # orange
    # ("#D2E4E1", "#207868"),   # green
    # ("#D1EFEC", "#1AAE9F"),   # mint
    # ("#F2D6F6", "#BD34D1"),   # purple
    ("#FDF3D3", "#F7C325"),   # yellow solid ModelSpider
    ("#1AAE9F", "#1AAE9F"),    # mint solid Cross-sight
    ("#D3455B", "#D3455B")    # red solid
]

# ==========================
# Assign colors
# ==========================
colors = []
edges = []

for i, m in enumerate(metrics):
    muted, stroke = palette[i % len(palette)]

    # ModelSpider + Cross-sight get SOLID stroke color
    if m in ["ModelSpider", "Cross-sight"]:
        colors.append(stroke)
        edges.append("black")
    else:
        colors.append(muted)   # muted fill
        edges.append(stroke)   # stroke edge

# ==========================
# Plotting
# ==========================
n_metrics = len(metrics)
n_datasets = df.shape[1]
x = np.arange(n_datasets)

total_width = 0.8
bar_width = total_width / n_metrics
offsets = (np.arange(n_metrics) - (n_metrics - 1)/2) * bar_width

fig, ax = plt.subplots(figsize=(11, 6))

for i, metric in enumerate(metrics):
    ax.bar(
        x + offsets[i],
        df.loc[metric].values,
        width=bar_width * 0.95,
        color=colors[i],
        edgecolor=edges[i],
        linewidth=1.0,
        label=metric
    )

# ==========================
# Aesthetics
# ==========================
ax.set_xticks(x)
ax.set_xticklabels(df.columns, fontsize=14, rotation=45, ha='right')
ax.axhline(0, color="black", linewidth=0.8)
# ax.set_ylim(-0.3, 1.05)
ax.set_ylim(-0.25, 1.05)
# ax.set_title("Mean Performance Across Methods", fontsize=18, pad=10)
ax.set_title("Blind Dataset Comparison", fontsize=18, pad=10)

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
plt.tight_layout()

# Save
plt.savefig("double_blind.png", dpi=150, bbox_inches='tight')
plt.show()
