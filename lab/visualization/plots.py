"""
Diagnostic plots — time series and multi-panel overview.
"""

import numpy as np


def time_series(dataset, coords=None, ax=None):
    """
    Plot q_i(t) for each degree of freedom.

    Parameters
    ----------
    coords : list of int, optional
        Which coordinates to plot.  Defaults to all.
    """
    import matplotlib.pyplot as plt
    if coords is None:
        coords = list(range(dataset.ndof))
    if ax is None:
        _, ax = plt.subplots()
    for i in coords:
        label = (dataset.hamiltonian.coords[i]
                 if i < len(dataset.hamiltonian.coords) else f"q{i}")
        ax.plot(dataset.t, dataset.q[:, i], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("coordinate")
    ax.legend()
    ax.set_title(dataset.hamiltonian.name)
    return ax


def momentum_series(dataset, coords=None, ax=None):
    """Plot p_i(t)."""
    import matplotlib.pyplot as plt
    if coords is None:
        coords = list(range(dataset.ndof))
    if ax is None:
        _, ax = plt.subplots()
    for i in coords:
        label = f"p_{dataset.hamiltonian.coords[i]}" if i < len(dataset.hamiltonian.coords) else f"p{i}"
        ax.plot(dataset.t, dataset.p[:, i], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("momentum")
    ax.legend()
    return ax


def overview(dataset):
    """
    Multi-panel diagnostic: coordinates, energy conservation, phase space.

    Returns the figure.
    """
    import matplotlib.pyplot as plt
    from lab.analysis.energy import energy_vs_time, conservation_error
    from lab.analysis.phase_space import plot_phase_portrait

    ndof = dataset.ndof
    n_rows = 2 + ndof
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows))

    time_series(dataset, ax=axes[0])
    axes[0].grid(True, alpha=0.3)

    conservation_error(dataset, ax=axes[1])
    axes[1].grid(True, alpha=0.3)

    for i in range(ndof):
        plot_phase_portrait(dataset, coord=i, ax=axes[2 + i])
        axes[2 + i].grid(True, alpha=0.3)

    fig.suptitle(f"{dataset.hamiltonian.name} — overview", fontsize=14)
    fig.tight_layout()
    return fig


def compare_runs(datasets, coord=0, ax=None):
    """
    Overlay q_i(t) from multiple DataSets (e.g. different ICs or
    integrators).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    for i, ds in enumerate(datasets):
        label = ds.metadata.get("label", f"run {i}")
        ax.plot(ds.t, ds.q[:, coord], label=label, alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel(ds.hamiltonian.coords[coord]
                  if coord < len(ds.hamiltonian.coords) else f"q{coord}")
    ax.legend(fontsize="small")
    return ax
