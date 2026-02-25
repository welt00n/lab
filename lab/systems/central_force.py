"""
Central force Hamiltonians — orbital mechanics.

All systems use polar coordinates (r, θ) with conjugate momenta (p_r, p_θ).
"""

import numpy as np
from lab.core.hamiltonian import Hamiltonian


def kepler(m=1.0, M=1.0, G=1.0):
    """
    Kepler problem (gravitational 1/r potential).

    q = [r, θ], p = [p_r, p_θ].
    H = p_r²/2m + p_θ²/(2mr²) − GMm/r
    """
    def kinetic(q, p):
        r = max(q[0], 1e-12)
        return p[0]**2 / (2*m) + p[1]**2 / (2*m*r**2)

    def potential(q, p):
        r = max(q[0], 1e-12)
        return -G * M * m / r

    def dHdp(q, p):
        r = max(q[0], 1e-12)
        return np.array([p[0] / m, p[1] / (m * r**2)])

    def dHdq(q, p):
        r = max(q[0], 1e-12)
        return np.array([
            -p[1]**2 / (m * r**3) + G * M * m / r**2,
            0.0,
        ])

    return Hamiltonian(
        ndof=2,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        dHdq=dHdq,
        name="Kepler orbit",
        coords=["r", "θ"],
        vis_hint={"type": "orbit", "coords": "polar"},
    )


def general_central(m=1.0, V_func=None, dVdr=None):
    """
    General central force with arbitrary radial potential V(r).

    Parameters
    ----------
    V_func : callable(r) -> float
        Radial potential.
    dVdr : callable(r) -> float, optional
        Derivative of V.  Falls back to numerical if omitted.
    """
    if V_func is None:
        V_func = lambda r: -1.0 / max(r, 1e-12)

    def kinetic(q, p):
        r = max(q[0], 1e-12)
        return p[0]**2 / (2*m) + p[1]**2 / (2*m*r**2)

    def potential(q, p):
        return V_func(max(q[0], 1e-12))

    def dHdp(q, p):
        r = max(q[0], 1e-12)
        return np.array([p[0] / m, p[1] / (m * r**2)])

    grad_q = None
    if dVdr is not None:
        def grad_q(q, p):
            r = max(q[0], 1e-12)
            return np.array([
                -p[1]**2 / (m * r**3) + dVdr(r),
                0.0,
            ])

    return Hamiltonian(
        ndof=2,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        dHdq=grad_q,
        name="central force",
        coords=["r", "θ"],
        vis_hint={"type": "orbit", "coords": "polar"},
    )


def precessing_orbit(m=1.0, M=1.0, G=1.0, eps=0.01):
    """
    Kepler + relativistic-like correction: V = -GMm/r − ε/r³.

    The extra 1/r³ term causes apsidal precession, mimicking the
    general-relativistic perihelion advance.
    """
    def V(r):
        r = max(r, 1e-12)
        return -G * M * m / r - eps / r**3

    def dV(r):
        r = max(r, 1e-12)
        return G * M * m / r**2 + 3 * eps / r**4

    return general_central(m=m, V_func=V, dVdr=dV)
