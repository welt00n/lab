"""
SweepGrid — generic 2D colour-grid primitive for outcome maps.

Knows nothing about coins, cubes, or physics. Takes an integer data
grid, a value→colour mapping, and axis metadata.
"""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors


def _to_rgb(data, value_to_color, nan_color=(0.85, 0.85, 0.85)):
    """Convert an integer grid to an RGB image via a colour map."""
    nh, na = data.shape
    rgb = np.full((nh, na, 3), nan_color)
    for val, hex_color in value_to_color.items():
        r, g, b = mcolors.to_rgb(hex_color)
        mask = data == val
        rgb[mask] = [r, g, b]
    return rgb


def create(data, value_to_color, extent, xlabel, ylabel, title, ax):
    """
    Draw an ``imshow`` outcome grid on *ax*.

    Returns the ``AxesImage`` handle for live updates via ``update()``.
    """
    from matplotlib.patches import Patch

    rgb = _to_rgb(data, value_to_color)
    img = ax.imshow(rgb, origin="lower", aspect="auto",
                    extent=extent, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return img


def add_legend(ax, value_to_color, labels):
    """Add a colour-patch legend to the axes."""
    from matplotlib.patches import Patch
    elems = [
        Patch(facecolor=c, edgecolor="gray", label=labels[v])
        for v, c in value_to_color.items()
    ]
    ax.legend(handles=elems, loc="upper right", fontsize=7)


def update(img, data, value_to_color):
    """Swap the image data for a live-updating grid."""
    rgb = _to_rgb(data, value_to_color)
    img.set_data(rgb)
