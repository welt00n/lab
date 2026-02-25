"""
Energy analysis — conservation diagnostics for Hamiltonian systems.
"""

import numpy as np


def energy_vs_time(dataset, ax=None):
    """Plot energy as a function of time."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(dataset.t, dataset.energy)
    ax.set_xlabel("time")
    ax.set_ylabel("H(q, p)")
    ax.set_title(f"{dataset.hamiltonian.name} — energy")
    return ax


def conservation_error(dataset, ax=None):
    """
    Plot relative energy deviation from initial value.

    For a symplectic integrator on a conservative system this should
    remain bounded (oscillating) rather than drifting.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    drift = dataset.energy_drift()
    ax.plot(dataset.t, drift)
    ax.set_xlabel("time")
    ax.set_ylabel("ΔE / E₀")
    ax.set_title(f"{dataset.hamiltonian.name} — energy conservation")
    ax.axhline(0, color="k", lw=0.5)
    return ax


def drift_rate(dataset):
    """
    Linear regression slope of energy vs time — measures systematic drift.

    Returns (slope, intercept) where slope ≈ 0 for a good symplectic run.
    """
    coeffs = np.polyfit(dataset.t, dataset.energy, 1)
    return float(coeffs[0]), float(coeffs[1])


def energy_components(dataset, ax=None):
    """Plot kinetic and potential energy separately."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    H = dataset.hamiltonian
    T = np.array([H.kinetic(dataset.q[i], dataset.p[i])
                  for i in range(dataset.nsteps)])
    V = np.array([H.potential(dataset.q[i], dataset.p[i])
                  for i in range(dataset.nsteps)])

    ax.plot(dataset.t, T, label="kinetic")
    ax.plot(dataset.t, V, label="potential")
    ax.plot(dataset.t, dataset.energy, label="total", ls="--", color="k")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend()
    ax.set_title(f"{dataset.hamiltonian.name} — energy components")
    return ax
