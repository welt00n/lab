"""
Charged particle Hamiltonians — electromagnetism.

Particles in uniform electric and/or magnetic fields.
Uses Cartesian coordinates in 3D: q = [x, y, z], p = [px, py, pz].

For a magnetic field, the canonical momentum is p = mv + qA, where A is
the vector potential.  The Hamiltonian approach automatically handles
this through the kinetic energy term T = |p − qA|² / 2m.
"""

import numpy as np
from lab.core.hamiltonian import Hamiltonian


def uniform_E(m=1.0, charge=1.0, E=None):
    """
    Charged particle in a uniform electric field.

    3 DOF Cartesian.  Default E points in +x.
    """
    if E is None:
        E = np.array([1.0, 0.0, 0.0])
    E = np.asarray(E, dtype=float)

    def kinetic(q, p):
        return np.dot(p, p) / (2 * m)

    def potential(q, p):
        return -charge * np.dot(E, q)

    def dHdp(q, p):
        return p / m

    def dHdq(q, p):
        return -charge * E

    return Hamiltonian(
        ndof=3,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        dHdq=dHdq,
        name="particle in uniform E",
        coords=["x", "y", "z"],
        vis_hint={"type": "particle_3d"},
    )


def uniform_B(m=1.0, charge=1.0, B=None):
    """
    Charged particle in a uniform magnetic field (cyclotron motion).

    Uses the symmetric gauge A = ½ B × r so that
    H = |p − qA(q)|² / 2m.

    The magnetic force is velocity-dependent and doesn't derive from a
    scalar potential; we handle it through the vector potential in the
    Hamiltonian.
    """
    if B is None:
        B = np.array([0.0, 0.0, 1.0])
    B = np.asarray(B, dtype=float)

    def A(q):
        return 0.5 * np.cross(B, q)

    def kinetic(q, p):
        v_canonical = p - charge * A(q)
        return np.dot(v_canonical, v_canonical) / (2 * m)

    def potential(q, p):
        return 0.0

    def dHdp(q, p):
        return (p - charge * A(q)) / m

    return Hamiltonian(
        ndof=3,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        name=f"cyclotron (B={np.linalg.norm(B):.2g})",
        coords=["x", "y", "z"],
        vis_hint={"type": "particle_3d"},
    )


def crossed_EB(m=1.0, charge=1.0, E=None, B=None):
    """
    Crossed E and B fields — produces E×B drift.

    Combines uniform electric and magnetic fields.
    """
    if E is None:
        E = np.array([1.0, 0.0, 0.0])
    if B is None:
        B = np.array([0.0, 0.0, 1.0])
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)

    def A(q):
        return 0.5 * np.cross(B, q)

    def kinetic(q, p):
        v_canonical = p - charge * A(q)
        return np.dot(v_canonical, v_canonical) / (2 * m)

    def potential(q, p):
        return -charge * np.dot(E, q)

    def dHdp(q, p):
        return (p - charge * A(q)) / m

    v_drift = np.cross(E, B) / np.dot(B, B)

    return Hamiltonian(
        ndof=3,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        name="E×B drift",
        coords=["x", "y", "z"],
        vis_hint={"type": "particle_3d", "drift_velocity": v_drift.tolist()},
    )
