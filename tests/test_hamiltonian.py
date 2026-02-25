"""Unit tests for Hamiltonian."""

import numpy as np
import pytest
from lab.core.hamiltonian import Hamiltonian


def _sho(m=1.0, k=1.0):
    """Simple harmonic oscillator for testing."""
    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / (2 * m),
        potential=lambda q, p: 0.5 * k * q[0]**2,
        dHdp=lambda q, p: np.array([p[0] / m]),
        dHdq=lambda q, p: np.array([k * q[0]]),
        name="test SHO",
        coords=["x"],
    )


class TestHamiltonianEvaluation:
    def test_total_energy(self):
        H = _sho(m=1.0, k=4.0)
        assert H.H([2.0], [0.0]) == pytest.approx(8.0)
        assert H.H([0.0], [3.0]) == pytest.approx(4.5)

    def test_kinetic_potential_sum(self):
        H = _sho()
        q, p = [1.5], [2.5]
        assert H.H(q, p) == pytest.approx(H.kinetic(q, p) + H.potential(q, p))

    def test_accepts_lists_and_arrays(self):
        H = _sho()
        assert H.H([1.0], [1.0]) == H.H(np.array([1.0]), np.array([1.0]))


class TestAnalyticalGradients:
    def test_grad_q(self):
        H = _sho(k=3.0)
        grad = H.grad_q([2.0], [0.0])
        np.testing.assert_allclose(grad, [6.0])

    def test_grad_p(self):
        H = _sho(m=2.0)
        grad = H.grad_p([0.0], [4.0])
        np.testing.assert_allclose(grad, [2.0])


class TestNumericalGradients:
    def test_numerical_grad_q_matches_analytical(self):
        H_analytical = _sho(k=3.0)
        H_numerical = Hamiltonian(
            ndof=1,
            kinetic=lambda q, p: p[0]**2 / 2,
            potential=lambda q, p: 1.5 * q[0]**2,
        )
        q, p = np.array([2.0]), np.array([1.0])
        np.testing.assert_allclose(
            H_numerical.grad_q(q, p),
            H_analytical.grad_q(q, p),
            atol=1e-5,
        )

    def test_numerical_grad_p_matches_analytical(self):
        H_numerical = Hamiltonian(
            ndof=1,
            kinetic=lambda q, p: p[0]**2 / 2,
            potential=lambda q, p: 0.5 * q[0]**2,
        )
        q, p = np.array([1.0]), np.array([3.0])
        np.testing.assert_allclose(
            H_numerical.grad_p(q, p), [3.0], atol=1e-5,
        )

    def test_multidof_numerical_grad(self):
        H = Hamiltonian(
            ndof=2,
            kinetic=lambda q, p: (p[0]**2 + p[1]**2) / 2,
            potential=lambda q, p: q[0]**2 + 3 * q[1]**2,
        )
        q, p = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        np.testing.assert_allclose(H.grad_q(q, p), [2.0, 12.0], atol=1e-5)
        np.testing.assert_allclose(H.grad_p(q, p), [3.0, 4.0], atol=1e-5)


class TestMetadata:
    def test_default_coords(self):
        H = Hamiltonian(ndof=3, kinetic=lambda q, p: 0, potential=lambda q, p: 0)
        assert H.coords == ["q0", "q1", "q2"]

    def test_repr(self):
        H = _sho()
        assert "test SHO" in repr(H)
        assert "ndof=1" in repr(H)
