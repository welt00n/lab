"""
DataSet — time-indexed simulation output with convenience methods.
"""

import numpy as np


class DataSet:
    """
    Stores the trajectory produced by an Experiment.

    Attributes
    ----------
    t : ndarray, shape (N,)
    q : ndarray, shape (N, ndof)
    p : ndarray, shape (N, ndof)
    energy : ndarray, shape (N,)
    hamiltonian : Hamiltonian
    metadata : dict
    """

    def __init__(self, t, q, p, energy, hamiltonian, metadata=None):
        self.t = np.asarray(t)
        self.q = np.asarray(q)
        self.p = np.asarray(p)
        self.energy = np.asarray(energy)
        self.hamiltonian = hamiltonian
        self.metadata = metadata or {}

    @property
    def ndof(self):
        return self.q.shape[1]

    @property
    def nsteps(self):
        return len(self.t)

    @property
    def duration(self):
        return self.t[-1] - self.t[0]

    def energy_drift(self):
        """Relative energy change from initial value."""
        E0 = self.energy[0]
        if abs(E0) < 1e-15:
            return self.energy - E0
        return (self.energy - E0) / abs(E0)

    def max_energy_error(self):
        return float(np.max(np.abs(self.energy_drift())))

    def coord(self, i):
        """Return time series for the i-th coordinate."""
        return self.q[:, i]

    def momentum(self, i):
        """Return time series for the i-th momentum."""
        return self.p[:, i]

    def plot_energy(self, ax=None):
        """Quick energy vs time plot."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.t, self.energy)
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        ax.set_title(f"{self.hamiltonian.name} — energy")
        return ax

    def plot_trajectory(self, i=0, ax=None):
        """Quick q_i(t) plot."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        label = self.hamiltonian.coords[i] if i < len(self.hamiltonian.coords) else f"q{i}"
        ax.plot(self.t, self.q[:, i], label=label)
        ax.set_xlabel("time")
        ax.set_ylabel(label)
        ax.legend()
        return ax

    def __repr__(self):
        return (f"DataSet({self.hamiltonian.name!r}, "
                f"nsteps={self.nsteps}, ndof={self.ndof})")
