"""
Hamiltonian ray optics — geometric optics as a Hamiltonian system.

In the geometric (short-wavelength) limit, light rays are governed by
the Hamiltonian:

    H = |p| / n(q)

where n(q) is the refractive index field.  The canonical equations
give Snell's law at interfaces and curved paths in graded-index media.

This module provides:
    - Refractive index field builders
    - Hamiltonian factory for ray tracing
    - Convenience functions for common optical elements
"""

import numpy as np
from lab.core.hamiltonian import Hamiltonian


def ray_hamiltonian(n_func, ndim=2, name="ray optics"):
    """
    Build a Hamiltonian for ray tracing in a medium with refractive
    index n(q).

    Parameters
    ----------
    n_func : callable(q) -> float
        Refractive index as a function of position.
    ndim : int
        Number of spatial dimensions (2 or 3).

    The Hamiltonian is H = |p| / n(q).
    Hamilton's equations:
        dq/dt = dH/dp = p / (|p| n(q))       (ray direction)
        dp/dt = -dH/dq = |p| ∇n / n(q)²      (bending)
    """
    def kinetic(q, p):
        return np.linalg.norm(p)

    def potential(q, p):
        return 0.0

    def H(q, p):
        n = n_func(q)
        if n < 1e-12:
            n = 1e-12
        return np.linalg.norm(p) / n

    def dHdp(q, p):
        p_mag = np.linalg.norm(p)
        if p_mag < 1e-15:
            return np.zeros_like(p)
        n = max(n_func(q), 1e-12)
        return p / (p_mag * n)

    def dHdq(q, p, eps=1e-7):
        p_mag = np.linalg.norm(p)
        if p_mag < 1e-15:
            return np.zeros_like(q)
        grad_n = np.zeros_like(q)
        for i in range(len(q)):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            grad_n[i] = (n_func(qp) - n_func(qm)) / (2 * eps)
        n = max(n_func(q), 1e-12)
        return -p_mag * grad_n / n**2

    ham = Hamiltonian(
        ndof=ndim,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        dHdq=dHdq,
        name=name,
        coords=["x", "y", "z"][:ndim],
        vis_hint={"type": "ray", "n_func": n_func},
    )
    ham._H_override = H
    ham.H = H
    return ham


# ===================================================================
# Refractive index field builders
# ===================================================================

def uniform_medium(n=1.0):
    """Constant refractive index."""
    return lambda q: n


def slab(n_inside=1.5, n_outside=1.0, x_start=1.0, x_end=2.0):
    """
    Dielectric slab between x_start and x_end.

    Uses a smooth tanh transition to avoid discontinuities that would
    require special handling in the integrator.
    """
    steepness = 200.0

    def n_func(q):
        left = 0.5 * (1 + np.tanh(steepness * (q[0] - x_start)))
        right = 0.5 * (1 + np.tanh(steepness * (x_end - q[0])))
        return n_outside + (n_inside - n_outside) * left * right

    return n_func


def spherical_lens(center, radius, n_lens=1.5, n_outside=1.0):
    """
    Circular/spherical lens at *center* with given *radius*.

    Smooth transition at the boundary.
    """
    center = np.asarray(center, dtype=float)
    steepness = 200.0

    def n_func(q):
        r = np.linalg.norm(np.asarray(q) - center)
        transition = 0.5 * (1 + np.tanh(steepness * (radius - r)))
        return n_outside + (n_lens - n_outside) * transition

    return n_func


def graded_index(n0=1.5, alpha=0.1, axis=1):
    """
    Graded-index (GRIN) medium: n(q) = n0 * (1 - α * q[axis]²).

    Models a parabolic index profile like in optical fibers.
    """
    def n_func(q):
        return max(n0 * (1 - alpha * q[axis]**2), 1.0)
    return n_func


def two_media(n1=1.0, n2=1.5, boundary_x=2.0):
    """Two half-spaces with a smooth boundary at x = boundary_x."""
    steepness = 200.0

    def n_func(q):
        return n1 + (n2 - n1) * 0.5 * (1 + np.tanh(steepness * (q[0] - boundary_x)))

    return n_func


# ===================================================================
# Convenience: launch a fan of rays
# ===================================================================

def launch_fan(n_func, origin, angles, speed=1.0, ndim=2):
    """
    Create a list of (q0, p0) pairs for a fan of rays from *origin*
    at the given angles (in radians from the x-axis).

    Returns list of (q0, p0) tuples.
    """
    origin = np.asarray(origin, dtype=float)
    ics = []
    for angle in angles:
        q0 = origin.copy()
        if ndim == 2:
            p0 = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            p0 = speed * np.array([np.cos(angle), np.sin(angle), 0.0])
        ics.append((q0, p0))
    return ics
