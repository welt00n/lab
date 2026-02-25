"""
Physical constraints for rigid body simulation.
"""

import numpy as np
from lab.core import quaternion as quat
from lab.systems.rigid_body.objects import PointParticle, RigidBody

_UP = np.array([0.0, 1.0, 0.0])


class FloorConstraint:
    """
    Rigid floor at y = 0 with restitution, Coulomb friction, and
    rolling resistance.

    For point particles: projects position, reflects y-momentum.
    For rigid bodies: finds lowest point, projects CM upward,
    applies contact impulse (normal + tangential friction) that
    affects both linear and angular momentum.

    Parameters
    ----------
    restitution : float
        Coefficient of restitution (0 = perfectly inelastic, 1 = elastic).
        At low contact velocities the effective restitution drops to zero
        to prevent infinite micro-bouncing.
    friction : float
        Coulomb friction coefficient. Tangential impulse is capped at
        mu * |j_normal|.
    rolling_resistance : float
        Fraction of angular momentum removed per contact event.
        Models energy lost to contact-zone deformation during rocking.
    """

    def __init__(self, restitution=1.0, friction=0.6, rolling_resistance=0.05):
        self.restitution = restitution
        self.friction = friction
        self.rolling_resistance = rolling_resistance

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

        in_contact = penetration < 0
        near_floor = penetration < 0.005

        if in_contact:
            body.position[1] -= penetration

            n = _UP
            omega_world = quat.rotate_vector(body.orientation, body.omega)
            v_contact = body.velocity + np.cross(omega_world, lever)
            v_n = np.dot(v_contact, n)

            if v_n < -1e-4:
                R = quat.to_rotation_matrix(body.orientation)
                safe_I = np.where(body.inertia > 1e-30, body.inertia, 1e-30)
                I_inv_body = np.diag(1.0 / safe_I)
                I_inv_world = R @ I_inv_body @ R.T

                r_cross_n = np.cross(lever, n)
                rotational_term = np.dot(
                    n, np.cross(I_inv_world @ r_cross_n, lever))
                inv_mass_eff = 1.0 / body.mass + rotational_term

                e_eff = self.restitution if abs(v_n) > 0.1 else 0.0
                j_n = -(1.0 + e_eff) * v_n / inv_mass_eff

                body.momentum += j_n * n
                tau_impulse_world = np.cross(lever, j_n * n)
                tau_impulse_body = quat.rotate_vector(
                    quat.conjugate(body.orientation), tau_impulse_world)
                body.angular_momentum += tau_impulse_body

                if self.friction > 0:
                    omega_world = quat.rotate_vector(
                        body.orientation, body.omega)
                    v_contact = body.velocity + np.cross(omega_world, lever)
                    v_tangential = v_contact - np.dot(v_contact, n) * n
                    v_t_mag = np.linalg.norm(v_tangential)

                    if v_t_mag > 1e-12:
                        t_hat = v_tangential / v_t_mag

                        r_cross_t = np.cross(lever, t_hat)
                        rot_term_t = np.dot(
                            t_hat,
                            np.cross(I_inv_world @ r_cross_t, lever))
                        inv_mass_eff_t = 1.0 / body.mass + rot_term_t
                        if inv_mass_eff_t < 1e-30:
                            inv_mass_eff_t = 1.0 / body.mass

                        j_t_desired = v_t_mag / inv_mass_eff_t
                        j_t = min(j_t_desired, self.friction * abs(j_n))

                        impulse = -j_t * t_hat
                        body.momentum += impulse
                        tau_fric_world = np.cross(lever, impulse)
                        tau_fric_body = quat.rotate_vector(
                            quat.conjugate(body.orientation), tau_fric_world)
                        body.angular_momentum += tau_fric_body

        if near_floor and self.rolling_resistance > 0:
            body.angular_momentum *= (1.0 - self.rolling_resistance)

            if body.kinetic_energy() < 1e-3:
                body.momentum[:] = 0.0
                body.angular_momentum[:] = 0.0
