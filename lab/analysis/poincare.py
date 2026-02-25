"""
Poincaré sections — stroboscopic and surface-of-section maps.
"""

import numpy as np


def stroboscopic(dataset, period, coord_x=0, coord_y=None, ax=None):
    """
    Stroboscopic Poincaré map: sample the trajectory at multiples of
    *period* and plot in the (q_i, p_i) plane.

    Parameters
    ----------
    dataset : DataSet
    period : float
        Sampling period (e.g. the drive period for a driven system).
    coord_x : int
        Degree of freedom for the horizontal axis.
    coord_y : int or None
        If None, plots q vs p for coord_x.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    indices = []
    t_sample = 0.0
    for i, t in enumerate(dataset.t):
        if t >= t_sample - 1e-12:
            indices.append(i)
            t_sample += period

    indices = np.array(indices)
    if coord_y is None:
        x = dataset.q[indices, coord_x]
        y = dataset.p[indices, coord_x]
        ylabel = f"p_{dataset.hamiltonian.coords[coord_x]}"
    else:
        x = dataset.q[indices, coord_x]
        y = dataset.q[indices, coord_y]
        ylabel = dataset.hamiltonian.coords[coord_y]

    ax.scatter(x, y, s=1, alpha=0.6)
    ax.set_xlabel(dataset.hamiltonian.coords[coord_x])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Poincaré section (T={period:.3g})")
    return ax


def surface_of_section(dataset, section_coord=0, section_value=0.0,
                       direction="positive", plot_coords=(1, None),
                       ax=None):
    """
    Surface-of-section: record crossings when q[section_coord] passes
    through *section_value*.

    Parameters
    ----------
    direction : "positive" | "negative" | "both"
        Which zero-crossing direction to record.
    plot_coords : tuple (i, j)
        If j is None, plot q_i vs p_i at each crossing.
        Otherwise plot q_i vs q_j.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    sc = section_coord
    q_sec = dataset.q[:, sc]

    crossings = []
    for i in range(1, len(q_sec)):
        prev = q_sec[i-1] - section_value
        curr = q_sec[i] - section_value
        if prev * curr < 0:
            if direction == "positive" and curr < 0:
                continue
            if direction == "negative" and curr > 0:
                continue
            frac = abs(prev) / (abs(prev) + abs(curr))
            q_interp = dataset.q[i-1] + frac * (dataset.q[i] - dataset.q[i-1])
            p_interp = dataset.p[i-1] + frac * (dataset.p[i] - dataset.p[i-1])
            crossings.append((q_interp, p_interp))

    if not crossings:
        return ax

    ci, cj = plot_coords
    xs, ys = [], []
    for q_c, p_c in crossings:
        xs.append(q_c[ci])
        if cj is None:
            ys.append(p_c[ci])
        else:
            ys.append(q_c[cj])

    ax.scatter(xs, ys, s=2, alpha=0.7)
    ax.set_xlabel(dataset.hamiltonian.coords[ci])
    ylabel = (f"p_{dataset.hamiltonian.coords[ci]}" if cj is None
              else dataset.hamiltonian.coords[cj])
    ax.set_ylabel(ylabel)
    ax.set_title("surface of section")
    return ax
