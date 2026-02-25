"""Tests for Hamiltonian ray optics."""

import numpy as np
import pytest
from lab.systems.ray_optics import (
    ray_hamiltonian, uniform_medium, slab, spherical_lens,
    graded_index, two_media, launch_fan,
)
from lab.core.experiment import Experiment


class TestRefractiveIndexBuilders:
    def test_uniform(self):
        n = uniform_medium(1.5)
        assert n(np.array([0, 0])) == 1.5
        assert n(np.array([100, -50])) == 1.5

    def test_slab_inside(self):
        n = slab(n_inside=2.0, n_outside=1.0, x_start=1.0, x_end=3.0)
        assert n(np.array([2.0, 0.0])) == pytest.approx(2.0, abs=0.01)

    def test_slab_outside(self):
        n = slab(n_inside=2.0, n_outside=1.0, x_start=1.0, x_end=3.0)
        assert n(np.array([0.0, 0.0])) == pytest.approx(1.0, abs=0.01)

    def test_spherical_lens_center(self):
        n = spherical_lens(center=[5, 0], radius=1.0, n_lens=1.5)
        assert n(np.array([5.0, 0.0])) == pytest.approx(1.5, abs=0.01)

    def test_graded_index_center(self):
        n = graded_index(n0=1.5, alpha=0.1, axis=1)
        assert n(np.array([0, 0])) == 1.5

    def test_two_media(self):
        n = two_media(n1=1.0, n2=2.0, boundary_x=5.0)
        assert n(np.array([0.0, 0.0])) == pytest.approx(1.0, abs=0.01)
        assert n(np.array([10.0, 0.0])) == pytest.approx(2.0, abs=0.01)


class TestRayHamiltonian:
    def test_straight_line_in_uniform(self):
        n = uniform_medium(1.0)
        H = ray_hamiltonian(n, ndim=2)
        data = Experiment(H, q0=[0, 0], p0=[1, 0],
                          dt=0.01, duration=5.0).run()
        np.testing.assert_allclose(data.q[-1, 1], 0.0, atol=0.01)
        assert data.q[-1, 0] > 4.5

    def test_ray_bends_in_gradient(self):
        n = graded_index(n0=1.5, alpha=0.05, axis=1)
        H = ray_hamiltonian(n, ndim=2)
        data = Experiment(H, q0=[0, 0.5], p0=[1, 0],
                          dt=0.01, duration=3.0, integrator="rk4").run()
        assert np.max(np.abs(data.q[:, 1])) > 0.01


class TestLaunchFan:
    def test_correct_count(self):
        n = uniform_medium(1.0)
        angles = np.linspace(-0.3, 0.3, 10)
        ics = launch_fan(n, origin=[0, 0], angles=angles)
        assert len(ics) == 10

    def test_ic_shapes(self):
        n = uniform_medium(1.0)
        ics = launch_fan(n, origin=[0, 0], angles=[0, 0.1], ndim=2)
        q0, p0 = ics[0]
        assert q0.shape == (2,)
        assert p0.shape == (2,)
