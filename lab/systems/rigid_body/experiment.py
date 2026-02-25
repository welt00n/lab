"""
RigidBodyExperiment — runs a rigid body simulation and produces
a DataSet compatible with lab's analysis and visualisation tools.
"""

import numpy as np

from lab.core.dataset import DataSet
from lab.core.hamiltonian import Hamiltonian
from lab.systems.rigid_body.objects import RigidBody
from lab.systems.rigid_body.world import World
from lab.systems.rigid_body.fields import GravityField
from lab.systems.rigid_body.constraints import FloorConstraint


class RigidBodyExperiment:
    """
    Wraps the rigid body World integrator and outputs a DataSet.

    The DataSet stores:
        q: [x, y, z, qw, qx, qy, qz]  (7-vector)
        p: [px, py, pz, Lx, Ly, Lz]    (6-vector)
    """

    def __init__(self, bodies, fields=None, constraints=None,
                 dt=0.001, duration=5.0, environment=None):
        self.bodies = bodies if isinstance(bodies, list) else [bodies]
        self.fields = fields or []
        self.constraints = constraints or []
        self.dt = dt
        self.duration = duration
        self.environment = environment

    def run(self, progress=False):
        world = World()
        for b in self.bodies:
            world.add_particle(b)
        if self.environment:
            self.environment.apply(world)
        for f in self.fields:
            world.add_field(f)
        for c in self.constraints:
            world.add_constraint(c)

        nsteps = int(np.ceil(self.duration / self.dt))
        body = self.bodies[0]

        t_list = [0.0]
        q_list = [self._get_q(body)]
        p_list = [self._get_p(body)]
        e_list = [world.total_energy()]

        for i in range(nsteps):
            world.step(self.dt)
            t_list.append(world.time)
            q_list.append(self._get_q(body))
            p_list.append(self._get_p(body))
            e_list.append(world.total_energy())

            if progress and i % max(1, nsteps // 20) == 0:
                print(f"  {100*i/nsteps:.0f}%")

        shape = body.shape if isinstance(body, RigidBody) else "particle"
        dims = body.dimensions if isinstance(body, RigidBody) else {}
        if shape == "cube":
            size = dims.get("side", 0.3)
        elif shape == "coin":
            size = dims.get("radius", 0.15)
        elif shape == "rod":
            size = dims.get("length", 1.0)
        else:
            size = 0.3

        H = Hamiltonian(
            ndof=7,
            kinetic=lambda q, p: 0,
            potential=lambda q, p: 0,
            name=f"rigid body ({shape})",
            coords=["x", "y", "z", "qw", "qx", "qy", "qz"],
            vis_hint={"type": "rigid_drop", "shape": shape, "size": size},
        )

        return DataSet(
            t=np.array(t_list),
            q=np.array(q_list),
            p=np.array(p_list),
            energy=np.array(e_list),
            hamiltonian=H,
            metadata={"dt": self.dt, "bodies": len(self.bodies)},
        )

    @staticmethod
    def _get_q(body):
        if isinstance(body, RigidBody):
            return np.concatenate([body.position, body.orientation])
        return np.concatenate([body.position, [1, 0, 0, 0]])

    @staticmethod
    def _get_p(body):
        if isinstance(body, RigidBody):
            return np.concatenate([body.momentum, body.angular_momentum])
        return np.concatenate([body.momentum, [0, 0, 0]])


def drop_cube(side=0.3, mass=1.0, height=2.0, g=9.81, restitution=0.8,
              friction=0.6, orientation=None, angular_momentum=None, **kw):
    body = RigidBody.cube(
        mass=mass, side=side,
        position=np.array([0.0, height, 0.0]),
        momentum=np.zeros(3),
        orientation=orientation,
        angular_momentum=angular_momentum,
    )
    return RigidBodyExperiment(
        bodies=[body],
        fields=[GravityField(g)],
        constraints=[FloorConstraint(restitution, friction=friction)],
        **kw,
    )


def drop_coin(radius=0.15, mass=1.0, height=2.0, g=9.81, restitution=0.8,
              friction=0.6, orientation=None, angular_momentum=None, **kw):
    body = RigidBody.coin(
        mass=mass, radius=radius,
        position=np.array([0.0, height, 0.0]),
        momentum=np.zeros(3),
        orientation=orientation,
        angular_momentum=angular_momentum,
    )
    return RigidBodyExperiment(
        bodies=[body],
        fields=[GravityField(g)],
        constraints=[FloorConstraint(restitution, friction=friction)],
        **kw,
    )


def drop_rod(length=1.0, mass=1.0, height=2.0, g=9.81, restitution=0.8,
             friction=0.6, orientation=None, angular_momentum=None, **kw):
    body = RigidBody.rod(
        mass=mass, length=length,
        position=np.array([0.0, height, 0.0]),
        momentum=np.zeros(3),
        orientation=orientation,
        angular_momentum=angular_momentum,
    )
    return RigidBodyExperiment(
        bodies=[body],
        fields=[GravityField(g)],
        constraints=[FloorConstraint(restitution, friction=friction)],
        **kw,
    )
