"""Unit tests for Experiment and DataSet."""

import numpy as np
import pytest
from lab.core.hamiltonian import Hamiltonian
from lab.core.experiment import Experiment
from lab.core.dataset import DataSet


def _sho():
    return Hamiltonian(
        ndof=1,
        kinetic=lambda q, p: p[0]**2 / 2,
        potential=lambda q, p: 2.0 * q[0]**2,
        dHdp=lambda q, p: np.array([p[0]]),
        dHdq=lambda q, p: np.array([4.0 * q[0]]),
        name="SHO",
    )


class TestExperiment:
    def test_returns_dataset(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=1.0).run()
        assert isinstance(data, DataSet)

    def test_correct_number_of_steps(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=1.0).run()
        assert data.nsteps == 101

    def test_time_starts_at_zero(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=1.0).run()
        assert data.t[0] == 0.0

    def test_time_ends_at_duration(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=1.0).run()
        assert data.t[-1] == pytest.approx(1.0, abs=0.01)

    def test_initial_conditions_recorded(self):
        data = Experiment(_sho(), q0=[3.0], p0=[5.0], dt=0.01, duration=0.1).run()
        assert data.q[0, 0] == pytest.approx(3.0)
        assert data.p[0, 0] == pytest.approx(5.0)

    def test_energy_recorded(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=1.0).run()
        assert data.energy[0] == pytest.approx(2.0)

    def test_integrator_selection_rk4(self):
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01,
                          duration=1.0, integrator="rk4").run()
        assert data.max_energy_error() < 1e-8

    def test_custom_integrator_callable(self):
        from lab.core.integrators import rk4
        data = Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01,
                          duration=0.5, integrator=rk4).run()
        assert isinstance(data, DataSet)


class TestDataSet:
    @pytest.fixture
    def dataset(self):
        return Experiment(_sho(), q0=[1.0], p0=[0.0], dt=0.01, duration=5.0).run()

    def test_ndof(self, dataset):
        assert dataset.ndof == 1

    def test_duration(self, dataset):
        assert dataset.duration == pytest.approx(5.0, abs=0.01)

    def test_coord(self, dataset):
        assert len(dataset.coord(0)) == dataset.nsteps

    def test_momentum(self, dataset):
        assert len(dataset.momentum(0)) == dataset.nsteps

    def test_energy_drift_bounded(self, dataset):
        assert dataset.max_energy_error() < 0.01

    def test_energy_drift_zero_energy(self):
        H = Hamiltonian(ndof=1, kinetic=lambda q, p: 0, potential=lambda q, p: 0)
        data = Experiment(H, q0=[0], p0=[0], dt=0.1, duration=1.0).run()
        drift = data.energy_drift()
        np.testing.assert_allclose(drift, 0.0)

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "SHO" in r
        assert "ndof=1" in r
