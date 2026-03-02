"""Tests for pre-built system Hamiltonians."""

import numpy as np
import pytest
from lab.core.experiment import Experiment


class TestOscillators:
    def test_harmonic_energy_conserved(self):
        from lab.systems.oscillators import harmonic
        H = harmonic(m=1.0, k=4.0)
        data = Experiment(H, q0=[1.0], p0=[0.0], dt=0.01, duration=10.0).run()
        assert data.max_energy_error() < 1e-3

    def test_harmonic_frequency(self):
        from lab.systems.oscillators import harmonic
        H = harmonic(m=1.0, k=4.0)
        data = Experiment(H, q0=[1.0], p0=[0.0], dt=0.005, duration=50.0).run()
        from lab.analysis.spectral import dominant_frequency
        freq = dominant_frequency(data, coord=0)
        expected = np.sqrt(4.0) / (2 * np.pi)
        assert freq == pytest.approx(expected, rel=0.05)

    def test_coupled_two_dof(self):
        from lab.systems.oscillators import coupled
        H = coupled()
        assert H.ndof == 2
        data = Experiment(H, q0=[1, 0], p0=[0, 0], dt=0.01, duration=5.0).run()
        assert data.ndof == 2

    def test_duffing_runs(self):
        from lab.systems.oscillators import duffing
        H = duffing()
        data = Experiment(H, q0=[0.5], p0=[0.0], dt=0.01, duration=5.0).run()
        assert data.nsteps > 0


class TestPendulums:
    def test_simple_pendulum_small_angle(self):
        """Small angle pendulum should approximate SHO."""
        from lab.systems.pendulums import simple_pendulum
        H = simple_pendulum(m=1, l=1, g=9.81)
        data = Experiment(H, q0=[0.1], p0=[0.0], dt=0.001, duration=50.0).run()
        expected_freq = np.sqrt(9.81) / (2 * np.pi)
        from lab.analysis.spectral import dominant_frequency
        freq = dominant_frequency(data, coord=0)
        assert freq == pytest.approx(expected_freq, rel=0.05)

    def test_double_pendulum_energy_rk4(self):
        from lab.systems.pendulums import double_pendulum
        H = double_pendulum()
        data = Experiment(H, q0=[1.0, 0.5], p0=[0, 0], dt=0.002,
                          duration=5.0, integrator="rk4").run()
        assert data.max_energy_error() < 1e-5

    def test_spherical_pendulum_runs(self):
        from lab.systems.pendulums import spherical_pendulum
        H = spherical_pendulum()
        data = Experiment(H, q0=[1.0, 0.0], p0=[0.0, 2.0],
                          dt=0.005, duration=5.0, integrator="rk4").run()
        assert data.nsteps > 0


class TestCentralForce:
    def test_kepler_circular_orbit(self):
        """Circular orbit should maintain constant r."""
        from lab.systems.central_force import kepler
        GM = 100.0
        r0 = 5.0
        v_circ = np.sqrt(GM / r0)
        p_theta = v_circ * r0
        H = kepler(m=1, M=100, G=1)
        data = Experiment(H, q0=[r0, 0.0], p0=[0.0, p_theta],
                          dt=0.001, duration=20.0).run()
        r = data.q[:, 0]
        assert np.std(r) / np.mean(r) < 0.01

    def test_kepler_energy_conserved(self):
        from lab.systems.central_force import kepler
        H = kepler(m=1, M=100, G=1)
        data = Experiment(H, q0=[5.0, 0.0], p0=[0.0, 20.0],
                          dt=0.001, duration=20.0).run()
        assert data.max_energy_error() < 1e-4


class TestCharged:
    def test_uniform_E_constant_acceleration(self):
        from lab.systems.charged import uniform_E
        H = uniform_E(m=1.0, charge=1.0, E=[1.0, 0.0, 0.0])
        data = Experiment(H, q0=[0, 0, 0], p0=[0, 0, 0],
                          dt=0.01, duration=2.0).run()
        x_final = data.q[-1, 0]
        expected = 0.5 * 1.0 * 2.0**2
        assert x_final == pytest.approx(expected, rel=0.01)

    def test_cyclotron_circular(self):
        """Cyclotron motion should produce bounded periodic x,y motion."""
        from lab.systems.charged import uniform_B
        H = uniform_B(m=1.0, charge=1.0, B=[0, 0, 1])
        data = Experiment(H, q0=[0, 0, 0], p0=[1, 0, 0],
                          dt=0.005, duration=10.0, integrator="rk4").run()
        x_range = np.max(data.q[:, 0]) - np.min(data.q[:, 0])
        y_range = np.max(data.q[:, 1]) - np.min(data.q[:, 1])
        assert x_range > 0.1
        assert abs(x_range - y_range) / max(x_range, y_range) < 0.2
        assert abs(data.q[:, 2]).max() < 0.01
