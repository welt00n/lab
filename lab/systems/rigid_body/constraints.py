"""
Physical constraints for rigid body simulation.
"""

import numpy as np
from lab.core import quaternion as quat
from lab.systems.rigid_body.objects import PointParticle, RigidBody


class FloorConstraint:
    """
    Rigid floor at y = 0 with restitution.

    For point particles: projects position, reflects y-momentum.
    For rigid bodies: finds lowest point, projects CM upward,
    applies contact impulse (linear + angular).
    """

    def __init__(self, restitution=1.0):
        self.restitution = restitution

    def enforce(self, body):
        if isinstance(body, RigidBody):
            self._enforce_rigid(body)
        elif isinstance(body, PointParticle):
            self._enforce_point(body)

    def _enforce_point(self, p):
        if p.position[1] < 0:
            p.position[1] = 0.0
            p.momentum[1] = -self.restitution * p.momentum[1]
            if abs(p.momentum[1]) < 1e-12:
                p.momentum[1] = 0.0

    def _enforce_rigid(self, body):
        world_pt, lever = body.lowest_point()
        penetration = world_pt[1]
        if penetration >= 0:
            return

        body.position[1] -= penetration

        n = np.array([0.0, 1.0, 0.0])
        omega_world = quat.rotate_vector(body.orientation, body.omega)
        v_contact = body.velocity + np.cross(omega_world, lever)
        v_n = np.dot(v_contact, n)

        if v_n >= 0:
            return

        R = quat.to_rotation_matrix(body.orientation)
        safe_I = np.where(body.inertia > 1e-30, body.inertia, 1e-30)
        I_inv_body = np.diag(1.0 / safe_I)
        I_inv_world = R @ I_inv_body @ R.T

        r_cross_n = np.cross(lever, n)
        rotational_term = np.dot(n, np.cross(I_inv_world @ r_cross_n, lever))
        inv_mass_eff = 1.0 / body.mass + rotational_term

        j = -(1.0 + self.restitution) * v_n / inv_mass_eff

        body.momentum += j * n
        tau_impulse_world = np.cross(lever, j * n)
        tau_impulse_body = quat.rotate_vector(
            quat.conjugate(body.orientation), tau_impulse_world)
        body.angular_momentum += tau_impulse_body

        if np.linalg.norm(body.momentum) < 1e-12:
            body.momentum[:] = 0.0
        if np.linalg.norm(body.angular_momentum) < 1e-12:
            body.angular_momentum[:] = 0.0
