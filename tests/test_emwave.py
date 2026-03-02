"""Tests for the FDTD electromagnetic wave solver."""

import numpy as np
import pytest
from lab.systems.emwave import (
    FDTDGrid1D, FDTDGrid2D, FDTDDataSet,
    gaussian_pulse, sinusoidal_source, point_source_2d,
)


class TestFDTDGrid1D:
    def test_initial_fields_zero(self):
        grid = FDTDGrid1D(nx=100)
        np.testing.assert_array_equal(grid.Ey, 0)
        np.testing.assert_array_equal(grid.Bz, 0)

    def test_step_advances_time(self):
        grid = FDTDGrid1D(nx=100)
        grid.step()
        assert grid.time > 0
        assert grid.step_count == 1

    def test_gaussian_pulse_propagates(self):
        grid = FDTDGrid1D(nx=200, dx=0.01)
        grid.add_source(gaussian_pulse(50, amplitude=1.0, width=10, delay=20))

        for _ in range(100):
            grid.step()

        assert np.max(np.abs(grid.Ey)) > 0.01

    def test_run_returns_dataset(self):
        grid = FDTDGrid1D(nx=100)
        grid.add_source(gaussian_pulse(30))
        data = grid.run(nsteps=50, snapshot_interval=10)
        assert isinstance(data, FDTDDataSet)
        assert data.ndim == 1
        assert data.nsteps == 5

    def test_pml_absorbs(self):
        grid_no_pml = FDTDGrid1D(nx=200, dx=0.01)
        grid_no_pml.add_source(gaussian_pulse(100, delay=30, width=10))

        grid_pml = FDTDGrid1D(nx=200, dx=0.01)
        grid_pml.add_source(gaussian_pulse(100, delay=30, width=10))
        grid_pml.enable_pml(width=30)

        for _ in range(300):
            grid_no_pml.step()
            grid_pml.step()

        energy_no_pml = np.sum(grid_no_pml.Ey**2)
        energy_pml = np.sum(grid_pml.Ey**2)
        assert energy_pml < energy_no_pml

    def test_material_slows_wave(self):
        grid = FDTDGrid1D(nx=400, dx=0.01)
        grid.set_material(200, 400, epsilon=4.0)
        grid.add_source(gaussian_pulse(50, delay=30, width=10))

        for _ in range(200):
            grid.step()

        peak_pos = np.argmax(np.abs(grid.Ey))
        assert peak_pos < 200


class TestFDTDGrid2D:
    def test_initial_fields_zero(self):
        grid = FDTDGrid2D(nx=50, ny=50)
        np.testing.assert_array_equal(grid.Ez, 0)

    def test_point_source_radiates(self):
        grid = FDTDGrid2D(nx=50, ny=50, dx=0.02)
        grid.add_source(point_source_2d(25, 25, frequency=5.0))

        for _ in range(100):
            grid.step()

        assert np.max(np.abs(grid.Ez)) > 0

    def test_run_returns_dataset(self):
        grid = FDTDGrid2D(nx=30, ny=30)
        grid.add_source(point_source_2d(15, 15, frequency=5.0))
        data = grid.run(nsteps=20, snapshot_interval=10)
        assert data.ndim == 2
        assert data.Ez is not None
