"""
Force fields for 3D rigid body simulation.

Each field implements: force(body), potential(body), and optionally torque(body).
Pairwise fields (N-body) set pairwise=True and implement force_on/potential_on.
"""

import numpy as np

_ZEROS3 = np.zeros(3)


class GravityField:
    """Uniform gravitational field in -y.  V = mgy."""
    pairwise = False

    def __init__(self, g=9.81):
        self.g = g

    def force(self, body):
        return np.array([0.0, -body.mass * self.g, 0.0])

    def torque(self, body):
        return _ZEROS3

    def potential(self, body):
        return body.mass * self.g * body.position[1]


class UniformElectricField:
    """Constant electric field.  F = qE, V = -qE·r."""
    pairwise = False

    def __init__(self, E):
        self.E = np.asarray(E, dtype=float)

    def force(self, body):
        return body.charge * self.E

    def torque(self, body):
        return _ZEROS3

    def potential(self, body):
        return -body.charge * np.dot(self.E, body.position)


class CoulombField:
    """Pairwise Coulomb interaction between charged bodies."""
    pairwise = True

    def __init__(self, k=8.9875517873681764e9):
        self.k = k

    def force_on(self, body, all_bodies):
        f = np.zeros(3)
        for other in all_bodies:
            if other is body:
                continue
            r = body.position - other.position
            dist = np.linalg.norm(r)
            if dist < 1e-12:
                continue
            f += self.k * body.charge * other.charge * r / dist**3
        return f

    def torque_on(self, body, all_bodies):
        return _ZEROS3

    def potential_on(self, body, all_bodies):
        v = 0.0
        for other in all_bodies:
            if other is body:
                continue
            dist = np.linalg.norm(body.position - other.position)
            if dist < 1e-12:
                continue
            v += self.k * body.charge * other.charge / dist
        return 0.5 * v


class DragField:
    """Linear drag: F = -b·v.  Non-conservative."""
    pairwise = False

    def __init__(self, b=0.0):
        self.b = b

    def force(self, body):
        return -self.b * body.velocity

    def torque(self, body):
        if hasattr(body, "omega"):
            return -self.b * body.omega
        return _ZEROS3

    def potential(self, body):
        return 0.0
