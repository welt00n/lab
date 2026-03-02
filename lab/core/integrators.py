"""
Time integrators for Hamiltonian systems.

Each integrator advances a State by dt given a Hamiltonian, returning
a new State.
"""

import numpy as np
from lab.core.state import State


def leapfrog(H, state, dt):
    """
    Symplectic Leapfrog (Störmer-Verlet) — preserves phase-space volume.

    Kick-drift-kick scheme:
        p += -dH/dq * dt/2
        q += +dH/dp * dt
        p += -dH/dq * dt/2
    """
    q, p = state.q.copy(), state.p.copy()
    half = dt / 2.0

    p -= H.grad_q(q, p) * half
    q += H.grad_p(q, p) * dt
    p -= H.grad_q(q, p) * half

    return State(q, p)


def rk4(H, state, dt):
    """
    Classical 4th-order Runge-Kutta.

    Not symplectic, but higher-order accuracy.  Good for non-conservative
    systems or when accuracy per step matters more than long-term stability.
    """
    q, p = state.q.copy(), state.p.copy()

    def derivs(q, p):
        dqdt = H.grad_p(q, p)
        dpdt = -H.grad_q(q, p)
        return dqdt, dpdt

    k1q, k1p = derivs(q, p)
    k2q, k2p = derivs(q + 0.5*dt*k1q, p + 0.5*dt*k1p)
    k3q, k3p = derivs(q + 0.5*dt*k2q, p + 0.5*dt*k2p)
    k4q, k4p = derivs(q + dt*k3q, p + dt*k3p)

    q_new = q + (dt / 6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new = p + (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)

    return State(q_new, p_new)


def rk45_adaptive(H, state, dt, tol=1e-8):
    """
    Adaptive RK45 (Dormand-Prince) step.

    Returns (new_state, dt_used, dt_next).
    If the error exceeds *tol*, the step is retried with a smaller dt.
    """
    safety = 0.9
    dt_min = dt * 1e-6

    while True:
        s4 = rk4(H, state, dt)

        half = dt / 2.0
        s_mid = rk4(H, state, half)
        s5 = rk4(H, s_mid, half)

        err = max(np.max(np.abs(s5.q - s4.q)),
                  np.max(np.abs(s5.p - s4.p)))

        if err < 1e-15:
            return s5, dt, dt * 2.0

        if err < tol or dt <= dt_min:
            dt_next = min(dt * safety * (tol / err) ** 0.2, dt * 5.0)
            return s5, dt, max(dt_next, dt_min)

        dt = max(dt * safety * (tol / err) ** 0.25, dt_min)


INTEGRATORS = {
    "leapfrog": leapfrog,
    "rk4": rk4,
}
