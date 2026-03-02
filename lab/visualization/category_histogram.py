"""
CategoryHistogram — generic discrete bar-chart primitive.

Knows nothing about heads, tails, faces, or physics. Takes category
IDs, colours, and labels.
"""

from __future__ import annotations

import numpy as np


def create(categories, colors, labels, ax, title="distribution"):
    """
    Create a bar chart for discrete outcome categories.

    Parameters
    ----------
    categories : list[int]
        Sorted category IDs.
    colors : dict[int, str]
        Category → hex colour.
    labels : dict[int, str]
        Category → display label.
    ax : matplotlib Axes
    title : str

    Returns
    -------
    bars : BarContainer
        Handle for live updates via ``update()``.
    """
    bar_colors = [colors[k] for k in categories]
    bar_labels = [labels[k] for k in categories]
    bars = ax.bar(range(len(categories)), [0] * len(categories),
                  color=bar_colors)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(bar_labels, fontsize=8, rotation=30)
    ax.set_ylabel("count")
    ax.set_title(title)
    return bars


def update(bars, data, categories):
    """
    Update bar heights from a results grid.

    Parameters
    ----------
    bars : BarContainer
    data : ndarray
        Full results grid (integer values).
    categories : list[int]
    """
    counts = [int(np.nansum(data == k)) for k in categories]
    for bar, c in zip(bars, counts):
        bar.set_height(c)
    ax = bars[0].axes
    mx = max(1, max(counts))
    ax.set_ylim(0, mx * 1.15)
