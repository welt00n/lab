"""
Rigid body world — holds bodies, fields, constraints and integrates
the system with a symplectic Leapfrog scheme.
"""

import numpy as np
from lab.core import quaternion as quat
from lab.systems.rigid_body.objects import RigidBody


class World:
    """
    Physics world for 3D rigid body simulation.

    Implements the kick-drift-kick Leapfrog integrator with quaternion
    orientation updates.
    """

    def __init__(self):
        self.particles = []
        self.fields = []
        self.constraints = []
        self.time = 0.0

    def add_particle(self, body):
        self.particles.append(body)

    def add_field(self, field):
        self.fields.append(field)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def net_force(self, body):
        f = np.zeros(3)
        for field in self.fields:
            if getattr(field, "pairwise", False):
                f += field.force_on(body, self.particles)
            else:
                f += field.force(body)
        return f

    def net_torque(self, body):
        t = np.zeros(3)
        for field in self.fields:
            if getattr(field, "pairwise", False):
                if hasattr(field, "torque_on"):
                    t += field.torque_on(body, self.particles)
            else:
                if hasattr(field, "torque"):
                    t += field.torque(body)
        return t

    def enforce_constraints(self, body):
        for constraint in self.constraints:
            constraint.enforce(body)

    def step(self, dt):
        half = dt / 2.0

        for body in self.particles:
            F = self.net_force(body)
            body.momentum += F * half

            is_rigid = isinstance(body, RigidBody)
            if is_rigid:
                tau = self.net_torque(body)
                body.angular_momentum += tau * half

            body.position += body.velocity * dt

            if is_rigid:
                omega = body.omega
                dq = quat.exp_map(omega, dt)
                body.orientation = quat.normalize(
                    quat.multiply(dq, body.orientation))

            self.enforce_constraints(body)

            F = self.net_force(body)
            body.momentum += F * half

            if is_rigid:
                tau = self.net_torque(body)
                body.angular_momentum += tau * half

        self.time += dt

    def energy(self, body):
        ke = body.kinetic_energy()
        pe = 0.0
        for field in self.fields:
            if getattr(field, "pairwise", False):
                pe += field.potential_on(body, self.particles)
            else:
                pe += field.potential(body)
        return ke + pe

    def total_energy(self):
        return sum(self.energy(b) for b in self.particles)
