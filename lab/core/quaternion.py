"""
Quaternion utilities for 3D rotation.

Convention: q = [w, x, y, z] where w is the scalar part.
All functions operate on plain numpy arrays of shape (4,).
"""

import numpy as np


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def multiply(q1, q2):
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def conjugate(q):
    """Quaternion conjugate (negates the vector part)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def normalize(q):
    """Return unit quaternion."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def rotate_vector(q, v):
    """Rotate vector v by unit quaternion q:  v' = q v q*."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    rotated = multiply(multiply(q, qv), conjugate(q))
    return rotated[1:]


def from_axis_angle(axis, angle):
    """Quaternion representing rotation of *angle* radians about *axis*."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def exp_map(omega, dt):
    """
    Quaternion exponential: rotation by angular velocity *omega* over
    time step *dt*.  Handles the zero-angular-velocity case gracefully.
    """
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    half = theta / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    q = normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
