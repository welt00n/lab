"""
Oscillator Hamiltonians — the bread and butter of classical mechanics.

Each factory function returns a Hamiltonian object ready to plug into
an Experiment.
"""

import numpy as np
from lab.core.hamiltonian import Hamiltonian


def harmonic(m=1.0, k=1.0):
    """Simple harmonic oscillator: H = p²/2m + kq²/2."""
    omega = np.sqrt(k / m)
    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * m),
        potential=lambda q, p: 0.5 * k * q[0]**2,
        dHdp=lambda q, p: np.array([p[0] / m]),
        dHdq=lambda q, p: np.array([k * q[0]]),
        name=f"harmonic oscillator (ω={omega:.3g})",
        coords=["x"],
        vis_hint={"type": "spring", "rest_length": 0.0},
    )


def damped(m=1.0, k=1.0, gamma=0.1):
    """
    Damped harmonic oscillator.

    Damping is non-conservative so we model it as an effective
    Hamiltonian with a velocity-dependent potential term.  The
    integrator won't conserve energy (correct behavior).
    """
    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * m),
        potential=lambda q, p: 0.5 * k * q[0]**2,
        dHdp=lambda q, p: np.array([p[0] / m]),
        dHdq=lambda q, p: np.array([k * q[0] + gamma * p[0] / m]),
        name=f"damped oscillator (γ={gamma})",
        coords=["x"],
    )


def driven(m=1.0, k=1.0, gamma=0.1, F0=1.0, omega_d=1.0):
    """
    Driven damped oscillator.

    Uses analytical gradients that include the driving force and damping.
    The driving force depends on time, so we thread the current time
    through a mutable container (the Hamiltonian's vis_hint dict).
    Call H.vis_hint["clock"] = t before each gradient evaluation, or
    use the dedicated experiment runner which handles this.
    """
    clock = {"t": 0.0}

    def dHdq(q, p):
        drive = -F0 * np.cos(omega_d * clock["t"])
        return np.array([k * q[0] + gamma * p[0] / m + drive])

    H = Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * m),
        potential=lambda q, p: 0.5 * k * q[0]**2,
        dHdp=lambda q, p: np.array([p[0] / m]),
        dHdq=dHdq,
        name=f"driven oscillator (F₀={F0}, ω_d={omega_d})",
        coords=["x"],
    )
    H.vis_hint["clock"] = clock
    return H


def coupled(m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """
    Two coupled oscillators connected by springs.

    q = [x1, x2], p = [p1, p2].
    V = ½k1·x1² + ½k2·x2² + ½kc·(x1 - x2)²
    """
    def kinetic(q, p):
        return p[0]**2 / (2*m1) + p[1]**2 / (2*m2)

    def potential(q, p):
        return (0.5*k1*q[0]**2 + 0.5*k2*q[1]**2
                + 0.5*kc*(q[0] - q[1])**2)

    def dHdq(q, p):
        return np.array([
            k1*q[0] + kc*(q[0] - q[1]),
            k2*q[1] - kc*(q[0] - q[1]),
        ])

    def dHdp(q, p):
        return np.array([p[0]/m1, p[1]/m2])

    return Hamiltonian(
        ndof=2,
        kinetic=kinetic,
        potential=potential,
        dHdq=dHdq,
        dHdp=dHdp,
        name="coupled oscillators",
        coords=["x₁", "x₂"],
        vis_hint={"type": "coupled_spring"},
    )


def duffing(m=1.0, alpha=1.0, beta=0.25):
    """
    Duffing oscillator: V = ½α·q² + ¼β·q⁴.

    Classic nonlinear oscillator exhibiting bistability for β < 0
    and hardening/softening behavior.
    """
    def potential(q, p):
        return 0.5*alpha*q[0]**2 + 0.25*beta*q[0]**4

    def dHdq(q, p):
        return np.array([alpha*q[0] + beta*q[0]**3])

    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2*m),
        potential=potential,
        dHdq=dHdq,
        dHdp=lambda q, p: np.array([p[0] / m]),
        name="Duffing oscillator",
        coords=["x"],
    )
