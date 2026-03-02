"""
The Hamiltonian — the universal interface for classical systems.

A Hamiltonian object encapsulates H(q, p) and provides the partial
derivatives needed by the integrators (dH/dq, dH/dp) either via
numerical finite differences or user-supplied analytical gradients.
"""

import numpy as np


class Hamiltonian:
    """
    Parameters
    ----------
    ndof : int
        Number of degrees of freedom.
    kinetic : callable(q, p) -> float
        Kinetic energy T(q, p).
    potential : callable(q, p) -> float
        Potential energy V(q, p).
    dHdq : callable(q, p) -> ndarray, optional
        Analytical gradient of H w.r.t. q.  Falls back to finite
        differences if not provided.
    dHdp : callable(q, p) -> ndarray, optional
        Analytical gradient of H w.r.t. p.
    name : str
        Human-readable name for plots and reports.
    coords : list of str, optional
        Names for the generalized coordinates (e.g. ["theta", "phi"]).
    vis_hint : dict, optional
        Hints for the visualisation layer (e.g. {"type": "pendulum", ...}).
    """

    def __init__(self, ndof, kinetic, potential, *,
                 dHdq=None, dHdp=None,
                 name="system", coords=None, vis_hint=None):
        self.ndof = ndof
        self._kinetic = kinetic
        self._potential = potential
        self._dHdq = dHdq
        self._dHdp = dHdp
        self.name = name
        self.coords = coords or [f"q{i}" for i in range(ndof)]
        self.vis_hint = vis_hint or {}

    def H(self, q, p):
        """Total energy."""
        q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
        return float(self._kinetic(q, p) + self._potential(q, p))

    def kinetic(self, q, p):
        return float(self._kinetic(np.asarray(q, dtype=float),
                                   np.asarray(p, dtype=float)))

    def potential(self, q, p):
        return float(self._potential(np.asarray(q, dtype=float),
                                     np.asarray(p, dtype=float)))

    def grad_q(self, q, p, eps=1e-8):
        """dH/dq — used for dp/dt = -dH/dq."""
        if self._dHdq is not None:
            return np.asarray(self._dHdq(q, p), dtype=float)
        return self._numerical_grad_q(q, p, eps)

    def grad_p(self, q, p, eps=1e-8):
        """dH/dp — used for dq/dt = +dH/dp."""
        if self._dHdp is not None:
            return np.asarray(self._dHdp(q, p), dtype=float)
        return self._numerical_grad_p(q, p, eps)

    def _numerical_grad_q(self, q, p, eps):
        q = np.asarray(q, dtype=float)
        p = np.asarray(p, dtype=float)
        grad = np.zeros_like(q)
        for i in range(len(q)):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            grad[i] = (self.H(qp, p) - self.H(qm, p)) / (2 * eps)
        return grad

    def _numerical_grad_p(self, q, p, eps):
        q = np.asarray(q, dtype=float)
        p = np.asarray(p, dtype=float)
        grad = np.zeros_like(p)
        for i in range(len(p)):
            pp, pm = p.copy(), p.copy()
            pp[i] += eps
            pm[i] -= eps
            grad[i] = (self.H(q, pp) - self.H(q, pm)) / (2 * eps)
        return grad

    def visualization_hint(self):
        return self.vis_hint

    def __repr__(self):
        return f"Hamiltonian({self.name!r}, ndof={self.ndof})"
