"""
Pendulum Hamiltonians.

Uses exact (non-linearized) Hamiltonians so the full nonlinear
dynamics are captured.
"""

import numpy as np
from lab.core.hamiltonian import Hamiltonian


def simple_pendulum(m=1.0, l=1.0, g=9.81):
    """
    Simple pendulum.

    q = [θ], p = [p_θ].
    H = p_θ² / (2 m l²) − m g l cos θ
    """
    ml2 = m * l**2

    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * ml2),
        potential=lambda q, p: -m * g * l * np.cos(q[0]),
        dHdp=lambda q, p: np.array([p[0] / ml2]),
        dHdq=lambda q, p: np.array([m * g * l * np.sin(q[0])]),
        name="simple pendulum",
        coords=["θ"],
        vis_hint={"type": "pendulum", "lengths": [l], "masses": [m]},
    )


def double_pendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """
    Double pendulum — canonical example of deterministic chaos.

    q = [θ₁, θ₂], p = [p₁, p₂].
    Uses the standard Lagrangian → Hamiltonian transformation.
    """
    def kinetic(q, p):
        t1, t2 = q[0], q[1]
        p1, p2 = p[0], p[1]
        delta = t1 - t2
        denom = m1 + m2 * np.sin(delta)**2

        C1 = (p1 * p2 * np.sin(delta)) / (l1 * l2 * denom)
        C2 = ((m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2
               - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(delta))
              * np.sin(2 * delta) / (2 * l1**2 * l2**2 * denom**2))

        t1dot = (l2 * p1 - l1 * p2 * np.cos(delta)) / (l1**2 * l2 * denom)
        t2dot = ((m1 + m2) * l1 * p2 - m2 * l2 * p1 * np.cos(delta)) / (m2 * l1 * l2**2 * denom)

        return 0.5 * (m1 + m2) * l1**2 * t1dot**2 + 0.5 * m2 * l2**2 * t2dot**2 + m2 * l1 * l2 * t1dot * t2dot * np.cos(delta)

    def potential(q, p):
        return (-(m1 + m2) * g * l1 * np.cos(q[0])
                - m2 * g * l2 * np.cos(q[1]))

    def _velocities(q, p):
        delta = q[0] - q[1]
        denom = m1 + m2 * np.sin(delta)**2
        t1dot = (l2 * p[0] - l1 * p[1] * np.cos(delta)) / (l1**2 * l2 * denom)
        t2dot = ((m1 + m2) * l1 * p[1] - m2 * l2 * p[0] * np.cos(delta)) / (m2 * l1 * l2**2 * denom)
        return t1dot, t2dot

    def dHdp(q, p):
        t1dot, t2dot = _velocities(q, p)
        return np.array([t1dot, t2dot])

    return Hamiltonian(
        ndof=2,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        name="double pendulum",
        coords=["θ₁", "θ₂"],
        vis_hint={"type": "double_pendulum",
                  "lengths": [l1, l2], "masses": [m1, m2]},
    )


def spherical_pendulum(m=1.0, l=1.0, g=9.81):
    """
    Spherical pendulum — pendulum free to swing in 3D.

    q = [θ, φ], p = [p_θ, p_φ].
    H = p_θ² / (2ml²) + p_φ² / (2ml² sin²θ) − mgl cos θ
    """
    ml2 = m * l**2

    def kinetic(q, p):
        theta = q[0]
        sin_t = np.sin(theta)
        if abs(sin_t) < 1e-12:
            sin_t = 1e-12
        return p[0]**2 / (2*ml2) + p[1]**2 / (2*ml2*sin_t**2)

    def potential(q, p):
        return -m * g * l * np.cos(q[0])

    def dHdp(q, p):
        sin_t = np.sin(q[0])
        if abs(sin_t) < 1e-12:
            sin_t = 1e-12
        return np.array([p[0] / ml2, p[1] / (ml2 * sin_t**2)])

    def dHdq(q, p):
        theta = q[0]
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        if abs(sin_t) < 1e-12:
            sin_t = 1e-12
        dHdt = -p[1]**2 * cos_t / (ml2 * sin_t**3) + m * g * l * sin_t
        return np.array([dHdt, 0.0])

    return Hamiltonian(
        ndof=2,
        kinetic=kinetic,
        potential=potential,
        dHdp=dHdp,
        dHdq=dHdq,
        name="spherical pendulum",
        coords=["θ", "φ"],
    )
