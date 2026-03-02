"""
Rigid body and point particle types for 3D simulation.
"""

import numpy as np
from lab.core import quaternion as quat


class PointParticle:
    """
    A point particle in 3D.

    Attributes
    ----------
    mass : float
    charge : float
    position : ndarray (3,)
    momentum : ndarray (3,)
    """

    def __init__(self, mass, position, momentum, charge=0.0):
        self.mass = float(mass)
        self.charge = float(charge)
        self.position = np.asarray(position, dtype=float)
        self.momentum = np.asarray(momentum, dtype=float)

    @property
    def velocity(self):
        return self.momentum / self.mass

    def kinetic_energy(self):
        return np.dot(self.momentum, self.momentum) / (2.0 * self.mass)

    def __repr__(self):
        return (f"PointParticle(mass={self.mass}, charge={self.charge}, "
                f"position={self.position}, momentum={self.momentum})")


class RigidBody:
    """
    A rigid body in 3D with translational + rotational state.

    Translational: position (3,), momentum (3,)
    Rotational:    orientation (4,) unit quaternion [w,x,y,z],
                   angular_momentum (3,) in body frame.

    The inertia tensor is diagonal in the body frame (principal moments).
    """

    def __init__(self, mass, inertia, position, momentum,
                 orientation=None, angular_momentum=None,
                 charge=0.0, shape="sphere", dimensions=None):
        self.mass = float(mass)
        self.charge = float(charge)
        self.inertia = np.asarray(inertia, dtype=float)
        self.position = np.asarray(position, dtype=float)
        self.momentum = np.asarray(momentum, dtype=float)
        self.orientation = (
            np.array(quat.IDENTITY)
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        self.angular_momentum = (
            np.zeros(3)
            if angular_momentum is None
            else np.asarray(angular_momentum, dtype=float)
        )
        self.shape = shape
        self.dimensions = dimensions or {}

    @property
    def velocity(self):
        return self.momentum / self.mass

    @property
    def omega(self):
        """Body-frame angular velocity: omega_i = L_i / I_i."""
        safe_I = np.where(self.inertia > 1e-30, self.inertia, 1e-30)
        return self.angular_momentum / safe_I

    def kinetic_energy(self):
        translational = np.dot(self.momentum, self.momentum) / (2.0 * self.mass)
        rotational = np.dot(
            self.angular_momentum,
            self.angular_momentum / np.where(
                self.inertia > 1e-30, self.inertia, 1e-30)
        ) / 2.0
        return translational + rotational

    def lowest_point(self):
        """
        Returns (world_point, lever_arm) of the body point with minimum y.
        lever_arm is the vector from CM to that point in the world frame.
        """
        body_pts = self._body_frame_extremes()
        best_world = None
        best_y = np.inf
        best_lever = None
        for bp in body_pts:
            wp = quat.rotate_vector(self.orientation, bp)
            world = self.position + wp
            if world[1] < best_y:
                best_y = world[1]
                best_world = world
                best_lever = wp
        return best_world, best_lever

    def _body_frame_extremes(self):
        if self.shape == "rod":
            half = self.dimensions["length"] / 2.0
            return [np.array([half, 0, 0]), np.array([-half, 0, 0])]

        if self.shape == "coin":
            r = self.dimensions["radius"]
            half_t = self.dimensions.get("thickness", 0.02) / 2.0
            pts = [
                np.array([0.0, half_t, 0.0]),
                np.array([0.0, -half_t, 0.0]),
            ]
            n_pts = 16
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append(np.array([r * np.cos(a), half_t, r * np.sin(a)]))
                pts.append(np.array([r * np.cos(a), -half_t, r * np.sin(a)]))
            return pts

        if self.shape == "cube":
            h = self.dimensions["side"] / 2.0
            signs = np.array([[sx, sy, sz]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            return [h * s for s in signs]

        return [np.zeros(3)]

    def min_rest_height(self):
        """Minimum CM height when resting on the most stable face."""
        pts = self._body_frame_extremes()
        return min(abs(bp[1]) for bp in pts)

    def mesh(self):
        """Return (vertices, faces) in the world frame for visualisation."""
        body_verts, faces = self._body_mesh()
        R = quat.to_rotation_matrix(self.orientation)
        world_verts = (R @ body_verts.T).T + self.position
        return world_verts, faces

    def _body_mesh(self):
        if self.shape == "rod":
            return _rod_mesh(self.dimensions["length"],
                             self.dimensions.get("radius", 0.02))
        if self.shape == "coin":
            return _coin_mesh(self.dimensions["radius"],
                              self.dimensions.get("thickness", 0.02))
        if self.shape == "cube":
            return _cube_mesh(self.dimensions["side"])
        return _cube_mesh(self.dimensions.get("side", 0.1))

    @classmethod
    def rod(cls, mass, length, position, momentum,
            orientation=None, angular_momentum=None, charge=0.0):
        eps = 1e-6
        I_perp = mass * length**2 / 12.0
        inertia = np.array([eps, I_perp, I_perp])
        return cls(mass, inertia, position, momentum,
                   orientation, angular_momentum, charge,
                   shape="rod", dimensions={"length": length, "radius": 0.02})

    @classmethod
    def coin(cls, mass, radius, position, momentum,
             orientation=None, angular_momentum=None, charge=0.0):
        I_diam = mass * radius**2 / 4.0
        I_axis = mass * radius**2 / 2.0
        thickness = radius * 0.144  # ~1.75 mm for a quarter-sized coin
        inertia = np.array([I_diam, I_axis, I_diam])
        return cls(mass, inertia, position, momentum,
                   orientation, angular_momentum, charge,
                   shape="coin", dimensions={"radius": radius,
                                             "thickness": thickness})

    @classmethod
    def cube(cls, mass, side, position, momentum,
             orientation=None, angular_momentum=None, charge=0.0):
        I_face = mass * side**2 / 6.0
        inertia = np.array([I_face, I_face, I_face])
        return cls(mass, inertia, position, momentum,
                   orientation, angular_momentum, charge,
                   shape="cube", dimensions={"side": side})

    def __repr__(self):
        return (f"RigidBody(shape={self.shape!r}, mass={self.mass}, "
                f"pos={self.position}, orientation={self.orientation})")


# ===================================================================
# Mesh primitives
# ===================================================================

def _rod_mesh(length, radius, n=8):
    half = length / 2.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cy, cz = radius * np.cos(angles), radius * np.sin(angles)
    verts = []
    for x in (-half, half):
        for i in range(n):
            verts.append([x, cy[i], cz[i]])
    verts = np.array(verts)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n)))
    faces.append(tuple(range(n, 2 * n)))
    return verts, faces


def _coin_mesh(radius, thickness, n=24):
    half_t = thickness / 2.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cz = radius * np.cos(angles), radius * np.sin(angles)
    verts = []
    for y in (-half_t, half_t):
        for i in range(n):
            verts.append([cx[i], y, cz[i]])
    verts = np.array(verts)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n)))
    faces.append(tuple(range(n, 2 * n)))
    return verts, faces


def _cube_mesh(side):
    h = side / 2.0
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ])
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (2, 3, 7, 6),
        (0, 3, 7, 4), (1, 2, 6, 5),
    ]
    return verts, faces
