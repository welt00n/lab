"""Tests for the rigid body system."""

import numpy as np
import pytest
from lab.core import quaternion as quat
from lab.systems.rigid_body import (
    PointParticle, RigidBody, World,
    GravityField, FloorConstraint,
    RigidBodyExperiment, drop_cube, drop_coin, drop_rod,
)


class TestPointParticle:
    def test_velocity(self):
        p = PointParticle(mass=2.0, position=[0, 0, 0], momentum=[4, 0, 0])
        np.testing.assert_allclose(p.velocity, [2, 0, 0])

    def test_kinetic_energy(self):
        p = PointParticle(mass=1.0, position=[0, 0, 0], momentum=[3, 4, 0])
        assert p.kinetic_energy() == pytest.approx(12.5)


class TestRigidBody:
    def test_cube_factory(self):
        b = RigidBody.cube(mass=1.0, side=0.3,
                           position=[0, 1, 0], momentum=[0, 0, 0])
        assert b.shape == "cube"
        assert b.mass == 1.0
        assert b.inertia.shape == (3,)

    def test_coin_factory(self):
        b = RigidBody.coin(mass=0.01, radius=0.05,
                           position=[0, 1, 0], momentum=[0, 0, 0])
        assert b.shape == "coin"
        I_diam = 0.01 * 0.05**2 / 4
        assert b.inertia[0] == pytest.approx(I_diam)

    def test_rod_factory(self):
        b = RigidBody.rod(mass=1.0, length=2.0,
                          position=[0, 1, 0], momentum=[0, 0, 0])
        assert b.shape == "rod"

    def test_lowest_point_cube_identity(self):
        b = RigidBody.cube(mass=1.0, side=1.0,
                           position=[0, 5, 0], momentum=[0, 0, 0])
        world_pt, lever = b.lowest_point()
        assert world_pt[1] == pytest.approx(4.5)

    def test_coin_flat_lowest_is_face(self):
        b = RigidBody.coin(mass=1.0, radius=0.1,
                           position=[0, 5, 0], momentum=[0, 0, 0])
        world_pt, lever = b.lowest_point()
        thickness = 0.02
        assert world_pt[1] == pytest.approx(5.0 - thickness / 2)

    def test_mesh_returns_verts_faces(self):
        b = RigidBody.cube(mass=1.0, side=0.3,
                           position=[0, 1, 0], momentum=[0, 0, 0])
        verts, faces = b.mesh()
        assert verts.shape[1] == 3
        assert len(faces) > 0


class TestWorld:
    def test_free_fall(self):
        world = World()
        p = PointParticle(mass=1.0, position=[0, 10, 0], momentum=[0, 0, 0])
        world.add_particle(p)
        world.add_field(GravityField(g=10.0))

        for _ in range(1000):
            world.step(0.001)

        v_expected = -10.0 * 1.0
        assert p.velocity[1] == pytest.approx(v_expected, rel=0.01)

    def test_floor_stops_particle(self):
        world = World()
        p = PointParticle(mass=1.0, position=[0, 0.5, 0], momentum=[0, 0, 0])
        world.add_particle(p)
        world.add_field(GravityField())
        world.add_constraint(FloorConstraint(restitution=0.0))

        for _ in range(5000):
            world.step(0.001)

        assert p.position[1] >= -0.001


class TestRigidBodyExperiment:
    def test_drop_cube_returns_dataset(self):
        exp = drop_cube(dt=0.001, duration=1.0)
        data = exp.run()
        assert data.nsteps > 0
        assert data.q.shape[1] == 7  # x,y,z,qw,qx,qy,qz

    def test_drop_coin_returns_dataset(self):
        exp = drop_coin(dt=0.001, duration=1.0)
        data = exp.run()
        assert data.nsteps > 0

    def test_drop_rod_returns_dataset(self):
        exp = drop_rod(dt=0.001, duration=1.0)
        data = exp.run()
        assert data.nsteps > 0

    def test_cube_hits_floor(self):
        exp = drop_cube(height=2.0, restitution=0.0, dt=0.001, duration=3.0)
        data = exp.run()
        final_y = data.q[-1, 1]
        assert final_y < 0.5
