"""
Tests for the experiment framework: base class, CoinDrop, CubeDrop.

Covers subclass config validation, grid building, batch sweep output,
and default argument generation.
"""

import math
import numpy as np
import pytest

from lab.core.rigid_body_jit import COIN, CUBE
from lab.experiments.base import DropExperiment
from lab.experiments.coin import CoinDrop
from lab.experiments.cube import CubeDrop

REQUIRED_ATTRS = (
    "shape", "shape_id", "angle_range", "colors", "labels",
    "settle_height", "body_color", "mesh",
)


# ===================================================================
# Subclass configuration
# ===================================================================

class TestCoinDropConfig:
    exp = CoinDrop()

    def test_has_all_required_attrs(self):
        for attr in REQUIRED_ATTRS:
            assert hasattr(self.exp, attr), f"Missing attribute: {attr}"

    def test_shape_id_matches_constant(self):
        assert self.exp.shape_id == COIN

    def test_shape_name(self):
        assert self.exp.shape == "coin"

    def test_angle_range_full_circle(self):
        lo, hi = self.exp.angle_range
        assert abs(hi - 2 * math.pi) < 1e-10

    def test_colors_and_labels_consistent(self):
        for outcome_id in self.exp.colors:
            assert outcome_id in self.exp.labels, (
                f"Outcome {outcome_id} in colors but not labels")

    def test_settle_height_positive(self):
        assert self.exp.settle_height > 0


class TestCubeDropConfig:
    exp = CubeDrop()

    def test_has_all_required_attrs(self):
        for attr in REQUIRED_ATTRS:
            assert hasattr(self.exp, attr), f"Missing attribute: {attr}"

    def test_shape_id_matches_constant(self):
        assert self.exp.shape_id == CUBE

    def test_shape_name(self):
        assert self.exp.shape == "cube"

    def test_six_outcomes(self):
        assert len(self.exp.colors) == 6
        assert len(self.exp.labels) == 6

    def test_colors_and_labels_consistent(self):
        for outcome_id in self.exp.colors:
            assert outcome_id in self.exp.labels

    def test_settle_height_positive(self):
        assert self.exp.settle_height > 0


# ===================================================================
# build_grid
# ===================================================================

class TestBuildGrid:
    @pytest.fixture
    def coin(self):
        return CoinDrop()

    def test_returns_two_arrays(self, coin):
        heights, angles = coin.build_grid(10, 20, 0.1, 5.0)
        assert isinstance(heights, np.ndarray)
        assert isinstance(angles, np.ndarray)

    def test_heights_shape(self, coin):
        heights, _ = coin.build_grid(10, 20, 0.1, 5.0)
        assert len(heights) == 10

    def test_angles_shape(self, coin):
        _, angles = coin.build_grid(10, 20, 0.1, 5.0)
        assert len(angles) == 20

    def test_heights_span(self, coin):
        heights, _ = coin.build_grid(10, 20, 0.1, 5.0)
        np.testing.assert_allclose(heights[0], 0.1)
        np.testing.assert_allclose(heights[-1], 5.0)

    def test_angles_within_range(self, coin):
        _, angles = coin.build_grid(10, 20, 0.1, 5.0)
        lo, hi = coin.angle_range
        assert angles[0] >= lo
        assert angles[-1] < hi

    def test_angles_not_endpoint_inclusive(self, coin):
        _, angles = coin.build_grid(10, 20, 0.1, 5.0)
        _, hi = coin.angle_range
        assert angles[-1] < hi


# ===================================================================
# sweep (small grid)
# ===================================================================

class TestSweep:
    def test_coin_returns_correct_shape(self):
        exp = CoinDrop()
        heights = np.linspace(0.5, 2.0, 3)
        angles = np.linspace(0, math.pi, 4, endpoint=False)
        results = exp.sweep(heights, angles)
        assert results.shape == (3, 4)

    def test_coin_valid_outcomes(self):
        exp = CoinDrop()
        heights = np.linspace(0.5, 2.0, 3)
        angles = np.linspace(0, math.pi, 4, endpoint=False)
        results = exp.sweep(heights, angles)
        valid = {-1, 0, 1}
        for val in results.flat:
            assert int(val) in valid, f"Unexpected outcome: {val}"

    def test_cube_returns_correct_shape(self):
        exp = CubeDrop()
        heights = np.linspace(0.5, 2.0, 3)
        angles = np.linspace(0, math.pi, 4, endpoint=False)
        results = exp.sweep(heights, angles)
        assert results.shape == (3, 4)

    def test_cube_valid_outcomes(self):
        exp = CubeDrop()
        heights = np.linspace(0.5, 2.0, 3)
        angles = np.linspace(0, math.pi, 4, endpoint=False)
        results = exp.sweep(heights, angles)
        valid = set(range(6))
        for val in results.flat:
            assert int(val) in valid, f"Unexpected outcome: {val}"

    def test_coin_flat_low_drop_is_heads(self):
        exp = CoinDrop()
        heights = np.array([0.02])
        angles = np.array([0.0])
        results = exp.sweep(heights, angles)
        assert results[0, 0] == 1  # heads (identity orientation)


# ===================================================================
# default_args
# ===================================================================

class TestDefaultArgs:
    def test_returns_dict(self):
        d = DropExperiment.default_args()
        assert isinstance(d, dict)

    def test_has_required_keys(self):
        d = DropExperiment.default_args()
        for key in ("nh", "na", "hmin", "hmax", "axis"):
            assert key in d

    def test_types(self):
        d = DropExperiment.default_args()
        assert isinstance(d["nh"], int)
        assert isinstance(d["na"], int)
        assert isinstance(d["hmin"], float)
        assert isinstance(d["hmax"], float)
        assert isinstance(d["axis"], str)
