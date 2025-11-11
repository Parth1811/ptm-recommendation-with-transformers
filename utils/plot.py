from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='darkgrid')


def plot_multiple(plots, title, save_path=None, figsize=None):
    """Plot multiple graphs in subplots.

    Args:
        plots: List of dicts with plot data and config. Each dict can contain:
            - 'x': x-axis data (optional, defaults to indices)
            - 'y': y-axis data (optional, defaults to indices)
            - 'title': subplot title
            - 'xlabel', 'ylabel': axis labels
            - Any other matplotlib plot() kwargs (color, linestyle, marker, etc.)
        figsize: Tuple (width, height) for figure size. Auto-calculated if None.
        save_path: Path to save figure. If None, displays instead.
    """
    n = len(plots)
    if n == 0:
        return

    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    for i, plot_spec in enumerate(plots):
        ax = axes[i]
        x = plot_spec.get('x')
        y = plot_spec.get('y')

        # Infer missing dimension
        if x is None and y is None:
            continue
        elif x is None:
            x = np.arange(len(y))
        elif y is None:
            y = np.arange(len(x))

        # Extract matplotlib config
        plot_kwargs = {k: v for k, v in plot_spec.items()
                      if k not in ['x', 'y', 'title', 'xlabel', 'ylabel', 'xscale', 'yscale']}

        ax.plot(x, y, **plot_kwargs)

        if 'title' in plot_spec:
            ax.set_title(plot_spec['title'])
        if 'xlabel' in plot_spec:
            ax.set_xlabel(plot_spec['xlabel'])
        if 'ylabel' in plot_spec:
            ax.set_ylabel(plot_spec['ylabel'])
        if 'xscale' in plot_spec:
            ax.set_xscale(plot_spec['xscale'])
        if 'yscale' in plot_spec:
            ax.set_yscale(plot_spec['yscale'])
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
