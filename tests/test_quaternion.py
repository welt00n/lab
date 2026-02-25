"""Unit tests for quaternion math."""

import numpy as np
import pytest
from lab.core import quaternion as quat


class TestMultiply:
    def test_identity_left(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quat.multiply(quat.IDENTITY, q)
        np.testing.assert_allclose(result, q)

    def test_identity_right(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quat.multiply(q, quat.IDENTITY)
        np.testing.assert_allclose(result, q)

    def test_inverse(self):
        q = quat.normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = quat.multiply(q, quat.conjugate(q))
        np.testing.assert_allclose(result, quat.IDENTITY, atol=1e-14)


class TestNormalize:
    def test_unit_quaternion(self):
        q = np.array([3.0, 4.0, 0.0, 0.0])
        n = quat.normalize(q)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-14

    def test_zero_returns_identity(self):
        n = quat.normalize(np.zeros(4))
        np.testing.assert_array_equal(n, quat.IDENTITY)


class TestRotateVector:
    def test_identity_rotation(self):
        v = np.array([1.0, 2.0, 3.0])
        result = quat.rotate_vector(quat.IDENTITY, v)
        np.testing.assert_allclose(result, v, atol=1e-14)

    def test_90_deg_about_z(self):
        q = quat.from_axis_angle([0, 0, 1], np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])
        result = quat.rotate_vector(q, v)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)

    def test_180_deg_about_y(self):
        q = quat.from_axis_angle([0, 1, 0], np.pi)
        v = np.array([1.0, 0.0, 0.0])
        result = quat.rotate_vector(q, v)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-14)

    def test_preserves_magnitude(self):
        q = quat.from_axis_angle([1, 1, 1], 1.23)
        v = np.array([3.0, -2.0, 7.0])
        result = quat.rotate_vector(q, v)
        assert abs(np.linalg.norm(result) - np.linalg.norm(v)) < 1e-12


class TestFromAxisAngle:
    def test_zero_angle(self):
        q = quat.from_axis_angle([1, 0, 0], 0.0)
        np.testing.assert_allclose(q, quat.IDENTITY, atol=1e-14)

    def test_full_rotation(self):
        q = quat.from_axis_angle([0, 0, 1], 2 * np.pi)
        v = np.array([1.0, 0.0, 0.0])
        result = quat.rotate_vector(q, v)
        np.testing.assert_allclose(result, v, atol=1e-12)


class TestExpMap:
    def test_zero_omega(self):
        q = quat.exp_map(np.zeros(3), 1.0)
        np.testing.assert_array_equal(q, quat.IDENTITY)

    def test_matches_axis_angle(self):
        omega = np.array([0.0, 0.0, 2.0])
        dt = 0.5
        q_exp = quat.exp_map(omega, dt)
        q_aa = quat.from_axis_angle([0, 0, 1], 1.0)
        np.testing.assert_allclose(q_exp, q_aa, atol=1e-14)


class TestToRotationMatrix:
    def test_identity_gives_I(self):
        R = quat.to_rotation_matrix(quat.IDENTITY)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_orthogonal(self):
        q = quat.from_axis_angle([1, 2, 3], 0.7)
        R = quat.to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_consistent_with_rotate_vector(self):
        q = quat.from_axis_angle([1, -1, 2], 1.1)
        v = np.array([4.0, -3.0, 2.0])
        R = quat.to_rotation_matrix(q)
        np.testing.assert_allclose(R @ v, quat.rotate_vector(q, v), atol=1e-12)
