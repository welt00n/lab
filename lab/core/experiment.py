"""
Experiment — the runner that ties Hamiltonian + integrator + initial
conditions together and produces a DataSet.
"""

import numpy as np

from lab.core.state import State
from lab.core.dataset import DataSet
from lab.core import integrators as _int


class Experiment:
    """
    Parameters
    ----------
    hamiltonian : Hamiltonian
    q0, p0 : array-like
        Initial conditions.
    dt : float
        Time step.
    duration : float
        Total simulation time.
    integrator : str or callable
        "leapfrog" (default, symplectic), "rk4", or a custom function
        with signature ``f(H, state, dt) -> State``.
    """

    def __init__(self, hamiltonian, q0, p0, dt=0.01, duration=10.0,
                 integrator="leapfrog"):
        self.H = hamiltonian
        self.q0 = np.atleast_1d(np.asarray(q0, dtype=float))
        self.p0 = np.atleast_1d(np.asarray(p0, dtype=float))
        self.dt = dt
        self.duration = duration

        if callable(integrator):
            self._step = integrator
        elif integrator == "adaptive":
            self._step = None
            self._adaptive = True
        else:
            self._step = _int.INTEGRATORS[integrator]
            self._adaptive = False

    def run(self, progress=False):
        """
        Run the simulation, return a DataSet.

        Parameters
        ----------
        progress : bool
            If True, print a crude progress indicator.
        """
        nsteps = int(np.ceil(self.duration / self.dt))
        state = State(self.q0, self.p0)
        ndof = state.ndof

        t_arr = np.zeros(nsteps + 1)
        q_arr = np.zeros((nsteps + 1, ndof))
        p_arr = np.zeros((nsteps + 1, ndof))
        E_arr = np.zeros(nsteps + 1)

        t_arr[0] = 0.0
        q_arr[0] = state.q
        p_arr[0] = state.p
        E_arr[0] = self.H.H(state.q, state.p)

        t = 0.0
        adaptive = getattr(self, "_adaptive", False)

        for i in range(1, nsteps + 1):
            if adaptive:
                state, dt_used, dt_next = _int.rk45_adaptive(
                    self.H, state, self.dt)
            else:
                state = self._step(self.H, state, self.dt)

            t += self.dt
            t_arr[i] = t
            q_arr[i] = state.q
            p_arr[i] = state.p
            E_arr[i] = self.H.H(state.q, state.p)

            if progress and i % max(1, nsteps // 20) == 0:
                print(f"  {100*i/nsteps:.0f}%")

        return DataSet(t_arr, q_arr, p_arr, E_arr, self.H,
                       metadata={"dt": self.dt, "duration": self.duration,
                                 "integrator": str(self._step)})
