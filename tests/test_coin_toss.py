"""Tests for the coin toss experiment."""

import numpy as np
import pytest
from lab.experiments.coin_toss import (
    toss_coin, classify_final_orientation, sweep_h_vs_angle,
    HEADS, TAILS, EDGE,
)
from lab.core import quaternion as quat


class TestClassifyOrientation:
    def test_identity_is_heads(self):
        assert classify_final_orientation(quat.IDENTITY) == HEADS

    def test_flipped_is_tails(self):
        q = quat.from_axis_angle([1, 0, 0], np.pi)
        assert classify_final_orientation(q) == TAILS

    def test_edge_is_edge(self):
        q = quat.from_axis_angle([1, 0, 0], np.pi / 2)
        assert classify_final_orientation(q) == EDGE


class TestTossCoin:
    def test_returns_valid_result(self):
        result = toss_coin(height=0.3, tilt_axis="x", tilt_angle=0.0,
                           dt=0.001, restitution=0.5)
        assert result in (HEADS, TAILS, EDGE)

    def test_flat_heads_gives_heads_low_height(self):
        result = toss_coin(height=0.1, tilt_axis="x", tilt_angle=0.0,
                           dt=0.001, restitution=0.5,
                           friction=0.0, rolling_resistance=0.0)
        assert result == HEADS

    def test_flat_tails_gives_tails_low_height(self):
        result = toss_coin(height=0.1, tilt_axis="x", tilt_angle=np.pi,
                           dt=0.001, restitution=0.3,
                           friction=0.0, rolling_resistance=0.0)
        assert result == TAILS

    def test_trajectory_recording(self):
        result, traj = toss_coin(height=0.5, tilt_axis="x", tilt_angle=0.3,
                                  dt=0.001, restitution=0.5,
                                  record_trajectory=True)
        assert result in (HEADS, TAILS, EDGE)
        assert "t" in traj
        assert "y" in traj
        assert "energy" in traj
        assert len(traj["t"]) > 10

    def test_deterministic(self):
        """Same inputs should produce same output."""
        r1 = toss_coin(height=0.5, tilt_axis="x", tilt_angle=0.5,
                        dt=0.001, restitution=0.5)
        r2 = toss_coin(height=0.5, tilt_axis="x", tilt_angle=0.5,
                        dt=0.001, restitution=0.5)
        assert r1 == r2


class TestSweep:
    def test_returns_correct_shape(self):
        heights = np.array([0.1, 0.2])
        angles = np.array([0.0, 1.0, 2.0])
        results = sweep_h_vs_angle(heights, angles, tilt_axis="x",
                                    dt=0.001, restitution=0.5)
        assert results.shape == (2, 3)

    def test_all_valid_outcomes(self):
        heights = np.array([0.2])
        angles = np.array([0.0, np.pi])
        results = sweep_h_vs_angle(heights, angles, tilt_axis="x",
                                    dt=0.001, restitution=0.5)
        for r in results.flat:
            assert r in (HEADS, TAILS, EDGE)
