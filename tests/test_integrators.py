"""Unit tests for integrators."""

import numpy as np
import pytest
from lab.core.hamiltonian import Hamiltonian
from lab.core.state import State
from lab.core.integrators import leapfrog, rk4, rk45_adaptive


def _sho(k=1.0, m=1.0):
    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * m),
        potential=lambda q, p: 0.5 * k * q[0]**2,
        dHdp=lambda q, p: np.array([p[0] / m]),
        dHdq=lambda q, p: np.array([k * q[0]]),
    )


class TestLeapfrog:
    def test_returns_state(self):
        H = _sho()
        s = leapfrog(H, State([1.0], [0.0]), 0.01)
        assert isinstance(s, State)
        assert s.ndof == 1

    def test_energy_bounded_over_many_steps(self):
        """Leapfrog should keep energy bounded (not drifting) for SHO."""
        H = _sho(k=4.0)
        s = State([2.0], [0.0])
        E0 = H.H(s.q, s.p)
        for _ in range(10_000):
            s = leapfrog(H, s, 0.01)
        E_final = H.H(s.q, s.p)
        assert abs(E_final - E0) / E0 < 1e-3

    def test_sho_period(self):
        """After one full period, position should return close to start."""
        k, m = 4.0, 1.0
        omega = np.sqrt(k / m)
        period = 2 * np.pi / omega
        H = _sho(k=k, m=m)
        s = State([1.0], [0.0])
        dt = 0.001
        nsteps = int(round(period / dt))
        for _ in range(nsteps):
            s = leapfrog(H, s, dt)
        assert abs(s.q[0] - 1.0) < 0.01

    def test_does_not_mutate_input(self):
        H = _sho()
        s = State([1.0], [0.0])
        q_before = s.q.copy()
        leapfrog(H, s, 0.01)
        np.testing.assert_array_equal(s.q, q_before)


class TestRK4:
    def test_higher_accuracy_than_leapfrog(self):
        """RK4 should have smaller per-step error than leapfrog."""
        H = _sho(k=4.0)
        s = State([1.0], [0.0])
        dt = 0.1

        s_lf = leapfrog(H, s, dt)
        s_rk = rk4(H, s, dt)

        q_exact = np.cos(2.0 * dt)
        err_lf = abs(s_lf.q[0] - q_exact)
        err_rk = abs(s_rk.q[0] - q_exact)
        assert err_rk < err_lf

    def test_energy_over_long_run(self):
        H = _sho(k=4.0)
        s = State([2.0], [0.0])
        E0 = H.H(s.q, s.p)
        for _ in range(5_000):
            s = rk4(H, s, 0.01)
        E_final = H.H(s.q, s.p)
        assert abs(E_final - E0) / E0 < 1e-6


class TestAdaptive:
    def test_returns_tuple(self):
        H = _sho()
        s = State([1.0], [0.0])
        result = rk45_adaptive(H, s, 0.1)
        assert len(result) == 3
        new_state, dt_used, dt_next = result
        assert isinstance(new_state, State)
        assert dt_used > 0
        assert dt_next > 0
