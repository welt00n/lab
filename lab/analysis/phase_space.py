"""
Phase-space analysis — 2D phase portraits and trajectory plotting.
"""

import numpy as np


def plot_phase_portrait(dataset, coord=0, ax=None, **kwargs):
    """
    Plot the trajectory in the (q_i, p_i) plane.

    Parameters
    ----------
    dataset : DataSet
    coord : int
        Which degree of freedom to plot.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    q = dataset.q[:, coord]
    p = dataset.p[:, coord]
    label = kwargs.pop("label", dataset.hamiltonian.name)
    ax.plot(q, p, label=label, **kwargs)
    cname = (dataset.hamiltonian.coords[coord]
             if coord < len(dataset.hamiltonian.coords) else f"q{coord}")
    ax.set_xlabel(cname)
    ax.set_ylabel(f"p_{cname}")
    ax.set_title(f"phase portrait — {cname}")
    ax.set_aspect("auto")
    return ax


def multi_trajectory(datasets, coord=0, ax=None, colormap="viridis"):
    """
    Overlay multiple trajectories in phase space.

    Parameters
    ----------
    datasets : list of DataSet
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap(colormap)
    n = len(datasets)
    for i, ds in enumerate(datasets):
        color = cmap(i / max(n - 1, 1))
        plot_phase_portrait(ds, coord=coord, ax=ax,
                            color=color, label=f"IC {i}")
    ax.legend(fontsize="small")
    return ax


def energy_contours(hamiltonian, q_range, p_range, nlevels=20,
                    coord=0, ax=None):
    """
    Draw contour lines of constant energy in the (q_i, p_i) plane.

    Only works cleanly for 1-DOF systems (or if other DOFs are held
    fixed at zero).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    q_vals = np.linspace(*q_range, 200)
    p_vals = np.linspace(*p_range, 200)
    Q, P = np.meshgrid(q_vals, p_vals)
    ndof = hamiltonian.ndof
    H_grid = np.zeros_like(Q)

    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            q_full = np.zeros(ndof)
            p_full = np.zeros(ndof)
            q_full[coord] = Q[i, j]
            p_full[coord] = P[i, j]
            H_grid[i, j] = hamiltonian.H(q_full, p_full)

    ax.contour(Q, P, H_grid, levels=nlevels, cmap="coolwarm")
    cname = (hamiltonian.coords[coord]
             if coord < len(hamiltonian.coords) else f"q{coord}")
    ax.set_xlabel(cname)
    ax.set_ylabel(f"p_{cname}")
    ax.set_title(f"energy contours — {hamiltonian.name}")
    return ax
