"""
Tests for the JIT-compiled rigid-body physics core.

Covers quaternion math, lowest-point calculations, classification,
physical constants, the batch stepper, and the warmup helper.
"""

import math
import numpy as np
import pytest

from lab.core.rigid_body_jit import (
    COIN, CUBE, ROD,
    COIN_RADIUS, COIN_HALF_THICK, COIN_MASS,
    CUBE_HALF_SIDE, CUBE_MASS,
    ROD_HALF_LEN, ROD_RAD, ROD_MASS,
    quat_normalize, quat_multiply, quat_conjugate,
    quat_rotate, quat_from_axis_angle, quat_exp_map,
    cross, dot,
    _lowest_coin, _lowest_cube, _lowest_rod, lowest_point,
    classify, get_mass, get_inertia,
    step_bodies, warmup,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)
PI = math.pi


# ===================================================================
# Quaternion math
# ===================================================================

class TestQuatNormalize:
    def test_unit_length_output(self):
        w, x, y, z = quat_normalize(3.0, 0.0, 4.0, 0.0)
        assert abs(w*w + x*x + y*y + z*z - 1.0) < 1e-12

    def test_zero_returns_identity(self):
        w, x, y, z = quat_normalize(0.0, 0.0, 0.0, 0.0)
        assert (w, x, y, z) == (1.0, 0.0, 0.0, 0.0)

    def test_already_unit(self):
        w, x, y, z = quat_normalize(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose([w, x, y, z], [1, 0, 0, 0], atol=1e-15)


class TestQuatMultiply:
    def test_identity_left(self):
        w, x, y, z = quat_multiply(1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5)
        np.testing.assert_allclose([w, x, y, z], [0.5, 0.5, 0.5, 0.5], atol=1e-12)

    def test_identity_right(self):
        w, x, y, z = quat_multiply(0.5, 0.5, 0.5, 0.5, 1, 0, 0, 0)
        np.testing.assert_allclose([w, x, y, z], [0.5, 0.5, 0.5, 0.5], atol=1e-12)

    def test_inverse_gives_identity(self):
        qw, qx, qy, qz = quat_from_axis_angle(0.0, 0.0, 1.0, 1.0)
        cw, cx, cy, cz = quat_conjugate(qw, qx, qy, qz)
        w, x, y, z = quat_multiply(qw, qx, qy, qz, cw, cx, cy, cz)
        np.testing.assert_allclose([w, x, y, z], [1, 0, 0, 0], atol=1e-12)


class TestQuatRotate:
    def test_identity_rotation(self):
        rx, ry, rz = quat_rotate(1, 0, 0, 0, 1.0, 2.0, 3.0)
        np.testing.assert_allclose([rx, ry, rz], [1, 2, 3], atol=1e-12)

    def test_90deg_about_z(self):
        qw, qx, qy, qz = quat_from_axis_angle(0.0, 0.0, 1.0, PI / 2)
        rx, ry, rz = quat_rotate(qw, qx, qy, qz, 1.0, 0.0, 0.0)
        np.testing.assert_allclose([rx, ry, rz], [0, 1, 0], atol=1e-6)

    def test_preserves_magnitude(self):
        qw, qx, qy, qz = quat_from_axis_angle(1.0, 1.0, 1.0, 0.73)
        rx, ry, rz = quat_rotate(qw, qx, qy, qz, 3.0, 4.0, 0.0)
        assert abs(rx*rx + ry*ry + rz*rz - 25.0) < 1e-10


class TestQuatFromAxisAngle:
    def test_zero_angle_gives_identity(self):
        w, x, y, z = quat_from_axis_angle(0.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose([w, x, y, z], [1, 0, 0, 0], atol=1e-12)

    def test_pi_about_z(self):
        w, x, y, z = quat_from_axis_angle(0.0, 0.0, 1.0, PI)
        np.testing.assert_allclose([w, x, y, z], [0, 0, 0, 1], atol=1e-6)

    def test_unit_quaternion(self):
        w, x, y, z = quat_from_axis_angle(1.0, 2.0, 3.0, 1.5)
        assert abs(w*w + x*x + y*y + z*z - 1.0) < 1e-12


class TestQuatExpMap:
    def test_zero_omega(self):
        w, x, y, z = quat_exp_map(0.0, 0.0, 0.0, 0.01)
        np.testing.assert_allclose([w, x, y, z], [1, 0, 0, 0], atol=1e-12)

    def test_matches_from_axis_angle(self):
        omega_mag = 5.0
        dt = 0.01
        angle = omega_mag * dt
        w1, x1, y1, z1 = quat_exp_map(0.0, omega_mag, 0.0, dt)
        w2, x2, y2, z2 = quat_from_axis_angle(0.0, 1.0, 0.0, angle)
        np.testing.assert_allclose([w1, x1, y1, z1],
                                   [w2, x2, y2, z2], atol=1e-10)


class TestCrossDot:
    def test_cross_basic(self):
        rx, ry, rz = cross(1, 0, 0, 0, 1, 0)
        np.testing.assert_allclose([rx, ry, rz], [0, 0, 1])

    def test_cross_anticommutative(self):
        ax, ay, az = cross(1, 2, 3, 4, 5, 6)
        bx, by, bz = cross(4, 5, 6, 1, 2, 3)
        np.testing.assert_allclose([ax, ay, az], [-bx, -by, -bz])

    def test_dot_basic(self):
        assert dot(1, 0, 0, 0, 1, 0) == 0.0

    def test_dot_self(self):
        assert abs(dot(3, 4, 0, 3, 4, 0) - 25.0) < 1e-12


# ===================================================================
# Lowest-point functions
# ===================================================================

class TestLowestCoin:
    def test_identity_flat(self):
        py = 1.0
        y_low, _, _, _ = _lowest_coin(*IDENTITY, py)
        assert abs(y_low - (py - COIN_HALF_THICK)) < 1e-10

    def test_90deg_tilt(self):
        py = 1.0
        qw, qx, qy, qz = quat_from_axis_angle(0.0, 0.0, 1.0, PI / 2)
        y_low, _, _, _ = _lowest_coin(qw, qx, qy, qz, py)
        assert abs(y_low - (py - COIN_RADIUS)) < 1e-4


class TestLowestCube:
    def test_identity_flat(self):
        py = 1.0
        y_low, _, _, _ = _lowest_cube(*IDENTITY, py)
        assert abs(y_low - (py - CUBE_HALF_SIDE)) < 1e-10

    def test_45deg_tilt_lower(self):
        py = 1.0
        qw, qx, qy, qz = quat_from_axis_angle(1.0, 0.0, 0.0, PI / 4)
        y_low, _, _, _ = _lowest_cube(qw, qx, qy, qz, py)
        y_flat, _, _, _ = _lowest_cube(*IDENTITY, py)
        assert y_low < y_flat


class TestLowestRod:
    def test_identity(self):
        py = 1.0
        y_low, _, _, _ = _lowest_rod(*IDENTITY, py)
        assert abs(y_low - (py - ROD_RAD)) < 1e-10


class TestLowestPointDispatch:
    def test_coin(self):
        y1, _, _, _ = lowest_point(COIN, *IDENTITY, 1.0)
        y2, _, _, _ = _lowest_coin(*IDENTITY, 1.0)
        assert y1 == y2

    def test_cube(self):
        y1, _, _, _ = lowest_point(CUBE, *IDENTITY, 1.0)
        y2, _, _, _ = _lowest_cube(*IDENTITY, 1.0)
        assert y1 == y2

    def test_rod(self):
        y1, _, _, _ = lowest_point(ROD, *IDENTITY, 1.0)
        y2, _, _, _ = _lowest_rod(*IDENTITY, 1.0)
        assert y1 == y2


# ===================================================================
# Classification
# ===================================================================

class TestClassify:
    def test_coin_identity_is_heads(self):
        assert classify(COIN, *IDENTITY) == 1

    def test_coin_flipped_is_tails(self):
        q = quat_from_axis_angle(1.0, 0.0, 0.0, PI)
        assert classify(COIN, *q) == -1

    def test_coin_edge(self):
        q = quat_from_axis_angle(1.0, 0.0, 0.0, PI / 2)
        assert classify(COIN, *q) == 0

    def test_cube_identity(self):
        result = classify(CUBE, *IDENTITY)
        assert result in range(6)

    def test_cube_90deg_rotations_differ(self):
        faces = set()
        for axis, angle in [
            ((1, 0, 0), 0), ((1, 0, 0), PI/2), ((1, 0, 0), PI),
            ((1, 0, 0), 3*PI/2), ((0, 0, 1), PI/2), ((0, 0, 1), -PI/2),
        ]:
            q = quat_from_axis_angle(*axis, angle)
            faces.add(classify(CUBE, *q))
        assert len(faces) == 6, f"Expected 6 distinct faces, got {faces}"

    def test_rod_identity(self):
        result = classify(ROD, *IDENTITY)
        assert result in {-1, 0, 1}


# ===================================================================
# Physical constants
# ===================================================================

class TestGetMass:
    @pytest.mark.parametrize("shape,expected", [
        (COIN, COIN_MASS), (CUBE, CUBE_MASS), (ROD, ROD_MASS),
    ])
    def test_returns_correct_mass(self, shape, expected):
        assert get_mass(shape) == expected

    def test_all_positive(self):
        for s in (COIN, CUBE, ROD):
            assert get_mass(s) > 0


class TestGetInertia:
    def test_all_positive(self):
        for s in (COIN, CUBE, ROD):
            ix, iy, iz = get_inertia(s)
            assert ix > 0 and iy > 0 and iz > 0

    def test_cube_symmetric(self):
        ix, iy, iz = get_inertia(CUBE)
        np.testing.assert_allclose(ix, iy, rtol=1e-10)
        np.testing.assert_allclose(iy, iz, rtol=1e-10)

    def test_coin_asymmetric(self):
        ix, iy, iz = get_inertia(COIN)
        assert abs(iy - ix) > 1e-15


# ===================================================================
# Batch stepper
# ===================================================================

class TestStepBodies:
    @staticmethod
    def _make_state(n, heights, shape_id=COIN):
        pos = np.zeros((n, 3), dtype=np.float64)
        mom = np.zeros((n, 3), dtype=np.float64)
        ori = np.zeros((n, 4), dtype=np.float64)
        amom = np.zeros((n, 3), dtype=np.float64)
        alive = np.ones(n, dtype=np.bool_)
        sc = np.zeros(n, dtype=np.int64)
        alive_idx = np.arange(n, dtype=np.int64)
        for i in range(n):
            pos[i, 1] = heights[i]
            ori[i] = [1, 0, 0, 0]
        return pos, mom, ori, amom, alive, sc, alive_idx

    def test_free_fall_reaches_floor(self):
        pos, mom, ori, amom, alive, sc, ai = self._make_state(1, [2.0])
        n_alive = 1
        for _ in range(1000):
            _, n_alive = step_bodies(
                pos, mom, ori, amom, alive, sc, ai, n_alive,
                COIN, 0.0005, 9.81, 0.6, 0.5, 0.05,
                5.0 * COIN_RADIUS, 500)
            if n_alive == 0:
                break
        assert pos[0, 1] < 0.1

    def test_settled_body_not_alive(self):
        pos, mom, ori, amom, alive, sc, ai = self._make_state(1, [0.05])
        n_alive = 1
        for _ in range(500):
            _, n_alive = step_bodies(
                pos, mom, ori, amom, alive, sc, ai, n_alive,
                COIN, 0.0005, 9.81, 0.6, 0.5, 0.05,
                5.0 * COIN_RADIUS, 500)
            if n_alive == 0:
                break
        assert not alive[0]

    def test_multiple_bodies_settle_independently(self):
        pos, mom, ori, amom, alive, sc, ai = self._make_state(
            3, [0.05, 1.0, 3.0])
        n_alive = 3
        settled_order = []
        for _ in range(2000):
            newly, n_alive = step_bodies(
                pos, mom, ori, amom, alive, sc, ai, n_alive,
                COIN, 0.0005, 9.81, 0.6, 0.5, 0.05,
                5.0 * COIN_RADIUS, 500)
            for k in newly:
                settled_order.append(int(k))
            if n_alive == 0:
                break
        assert len(settled_order) == 3
        assert settled_order[0] == 0  # lowest height settles first

    def test_body_does_not_gain_height_after_settling(self):
        pos, mom, ori, amom, alive, sc, ai = self._make_state(1, [0.5])
        n_alive = 1
        for _ in range(2000):
            _, n_alive = step_bodies(
                pos, mom, ori, amom, alive, sc, ai, n_alive,
                COIN, 0.0005, 9.81, 0.6, 0.5, 0.05,
                5.0 * COIN_RADIUS, 500)
            if n_alive == 0:
                break
        settled_h = pos[0, 1]
        assert settled_h < 0.1
        assert settled_h >= 0.0


# ===================================================================
# Warmup
# ===================================================================

class TestWarmup:
    @pytest.mark.parametrize("shape", [COIN, CUBE, ROD])
    def test_runs_without_error(self, shape):
        warmup(shape)
