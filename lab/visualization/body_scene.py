"""
BodyScene — generic 3D rigid-body renderer.

Supports two backends:
  - PyVista / VTK / OpenGL (fast, GPU-accelerated polygon rendering)
  - Matplotlib Poly3DCollection fallback (slow, CPU depth-sorting)

Knows nothing about coins or cubes. Receives a mesh template and
colours from the experiment.
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Visual mesh dimensions (display space, not physics)
VIS_COIN_RADIUS = 0.15
VIS_COIN_HT = 0.01
VIS_CUBE_HALF = 0.15


# ================================================================
# Mesh templates
# ================================================================

def coin_mesh_pyvista():
    return pv.Cylinder(
        radius=VIS_COIN_RADIUS,
        height=2 * VIS_COIN_HT,
        direction=(0, 0, 1),
        center=(0, 0, 0),
        resolution=20,
        capping=True,
    )


def cube_mesh_pyvista():
    h = VIS_CUBE_HALF
    return pv.Box(bounds=(-h, h, -h, h, -h, h))


def coin_mesh_mpl():
    """Coin polygon faces for matplotlib Poly3DCollection."""
    r, ht = VIS_COIN_RADIUS, VIS_COIN_HT
    n = 16
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cz = r * np.cos(theta), r * np.sin(theta)
    top = np.column_stack([cx, np.full(n, ht), cz])
    bot = np.column_stack([cx, np.full(n, -ht), cz])
    faces = [top, bot[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([top[i], top[j], bot[j], bot[i]]))
    return faces


def cube_mesh_mpl():
    """Cube polygon faces for matplotlib Poly3DCollection."""
    h = VIS_CUBE_HALF
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]])
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
           [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
    return [v[f] for f in idx]


def _get_mesh_mpl(mesh_name):
    if mesh_name == "coin":
        return coin_mesh_mpl()
    return cube_mesh_mpl()


def _get_mesh_pyvista(mesh_name):
    if mesh_name == "coin":
        return coin_mesh_pyvista()
    return cube_mesh_pyvista()


# ================================================================
# Quaternion rotate for mesh transform (pure Python, no Numba)
# ================================================================

def _qr_np(qw, qx, qy, qz, vx, vy, vz):
    aw = -qx * vx - qy * vy - qz * vz
    ax = qw * vx + qy * vz - qz * vy
    ay = qw * vy - qx * vz + qz * vx
    az = qw * vz + qx * vy - qy * vx
    rx = -aw * qx + ax * qw - ay * qz + az * qy
    ry = -aw * qy + ax * qz + ay * qw - az * qx
    rz = -aw * qz - ax * qy + ay * qx + az * qw
    return rx, ry, rz


def _transform_mesh(faces, qw, qx, qy, qz, px, py, pz):
    """Rotate and translate polygon face arrays."""
    out = []
    for f in faces:
        rotated = np.empty_like(f)
        for i in range(len(f)):
            rx, ry, rz = _qr_np(qw, qx, qy, qz,
                                 f[i, 0], f[i, 1], f[i, 2])
            rotated[i] = [px + rx, pz + rz, py + ry]
        out.append(rotated)
    return out


# ================================================================
# VTK user matrix
# ================================================================

def _build_user_matrix(qw, qx, qy, qz, px, py, pz, ox=0.0, oz=0.0):
    """Physics state -> VTK 4x4 matrix (y-up physics -> z-up display)."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    return np.array([
        [r00, r02, r01, px + ox],
        [r20, r22, r21, pz + oz],
        [r10, r12, r11, py],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ================================================================
# BodyScene — unified 3D renderer
# ================================================================

class BodyScene:
    """
    3D scene for N rigid bodies on a floor plane.

    Automatically uses PyVista (GPU) when available, falling back
    to matplotlib Poly3DCollection (CPU).
    """

    def __init__(self, mesh_name, N, offsets, max_height,
                 body_color="#7799bb", title="drop scene",
                 use_pyvista=None):
        self.N = N
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.max_height = max_height
        self.mesh_name = mesh_name
        self.body_color = body_color
        self._settled = set()

        if use_pyvista is None:
            use_pyvista = HAS_PYVISTA
        self._use_pyvista = use_pyvista

        if self._use_pyvista:
            self._init_pyvista(title)
        else:
            self._pyvista_plotter = None

    # ---- PyVista backend ----

    def _init_pyvista(self, title):
        template = _get_mesh_pyvista(self.mesh_name)
        from matplotlib.colors import to_rgb
        rgb = to_rgb(self.body_color)
        alpha = 0.78 if self.N > 100 else 0.88

        self._pyvista_plotter = pv.Plotter(title=title,
                                           window_size=(900, 700))
        self._pyvista_plotter.set_background("white")
        try:
            self._pyvista_plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass

        xr = self.offsets[:, 0]
        yr = self.offsets[:, 1]
        pad = 0.5
        floor = pv.Plane(
            center=((xr.min() + xr.max()) / 2,
                    (yr.min() + yr.max()) / 2, 0),
            direction=(0, 0, 1),
            i_size=float(xr.max() - xr.min()) + 2 * pad,
            j_size=float(yr.max() - yr.min()) + 2 * pad,
        )
        self._pyvista_plotter.add_mesh(floor, color="#cccccc", opacity=0.35)

        self._actors = []
        for _ in range(self.N):
            actor = self._pyvista_plotter.add_mesh(
                template.copy(), color=rgb, opacity=alpha,
                smooth_shading=True)
            self._actors.append(actor)

        cx = (xr.min() + xr.max()) / 2
        cy = (yr.min() + yr.max()) / 2
        span = max(float(xr.max() - xr.min()),
                   float(yr.max() - yr.min()), 1.0)
        dist = span * 1.5 + self.max_height * 0.4
        self._pyvista_plotter.camera_position = [
            (cx - dist * 0.65, cy - dist * 0.65,
             self.max_height * 0.55),
            (cx, cy, self.max_height * 0.20),
            (0, 0, 1),
        ]

        self._title_text = title
        self._title_actor = self._pyvista_plotter.add_text(
            title, position="upper_left", font_size=10, color="black")

    def show_nonblocking(self):
        if self._use_pyvista:
            self._pyvista_plotter.show(interactive_update=True)

    def update_all(self, positions, orientations, alive=None):
        """Set transforms for all alive bodies."""
        if self._use_pyvista:
            for k in range(self.N):
                if k in self._settled:
                    continue
                qw, qx, qy, qz = orientations[k]
                px, py, pz = positions[k]
                ox, oz = self.offsets[k]
                self._actors[k].user_matrix = _build_user_matrix(
                    qw, qx, qy, qz, px, py, pz, ox, oz)

    def mark_settled(self, k, color=None):
        self._settled.add(k)
        if color is not None and self._use_pyvista:
            from matplotlib.colors import to_rgb
            r, g, b = to_rgb(color)
            self._actors[k].GetProperty().SetColor(r, g, b)

    def set_title(self, text):
        if not self._use_pyvista:
            return
        if text != self._title_text:
            self._pyvista_plotter.remove_actor(self._title_actor)
            self._title_actor = self._pyvista_plotter.add_text(
                text, position="upper_left", font_size=10, color="black")
            self._title_text = text

    def render(self):
        if not self._use_pyvista:
            return
        try:
            iren = getattr(self._pyvista_plotter, "iren", None)
            if iren is not None:
                iren.process_events()
            self._pyvista_plotter.render()
        except Exception:
            pass

    def reset(self):
        self._settled.clear()
        if self._use_pyvista:
            from matplotlib.colors import to_rgb
            r, g, b = to_rgb(self.body_color)
            for actor in self._actors:
                actor.GetProperty().SetColor(r, g, b)

    def close(self):
        if self._use_pyvista and self._pyvista_plotter is not None:
            try:
                self._pyvista_plotter.close()
            except Exception:
                pass

    # ---- Matplotlib fallback helpers ----

    def init_mpl_scene(self, ax, scene_spacing):
        """Set up a matplotlib 3D axes with floor and mesh collections."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        nh_range = self.offsets[:, 1]
        na_range = self.offsets[:, 0]
        scene_w = (na_range.max() - na_range.min()) / 2 + 0.3
        scene_d = (nh_range.max() - nh_range.min()) / 2 + 0.3

        ax.set_xlim(-scene_w, scene_w)
        ax.set_ylim(-scene_d, scene_d)
        ax.set_zlim(-0.05, self.max_height * 1.1)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("z", fontsize=7)
        ax.set_zlabel("height (m)", fontsize=8)
        ax.view_init(elev=18, azim=-60)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.zaxis.set_tick_params(labelsize=7)

        fx = np.array([-scene_w, scene_w, scene_w, -scene_w])
        fz = np.array([-scene_d, -scene_d, scene_d, scene_d])
        fy = np.zeros(4)
        floor_poly = Poly3DCollection(
            [list(zip(fx, fz, fy))],
            alpha=0.15, facecolors="#888888",
            edgecolors="#aaaaaa", linewidths=0.5)
        ax.add_collection3d(floor_poly)

        ref_mesh = _get_mesh_mpl(self.mesh_name)
        edge_lw = 0.3 if self.N > 50 else 0.4
        mesh_alpha = 0.75 if self.N > 100 else 0.85

        self._mpl_ref_mesh = ref_mesh
        self._mpl_collections = []

        return ref_mesh, edge_lw, mesh_alpha

    def create_mpl_bodies(self, ax, positions, orientations,
                          edge_lw, mesh_alpha):
        """Create initial Poly3DCollection for each body."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self._mpl_collections = []
        for k in range(self.N):
            ox, oz = self.offsets[k]
            faces = _transform_mesh(
                self._mpl_ref_mesh,
                orientations[k, 0], orientations[k, 1],
                orientations[k, 2], orientations[k, 3],
                ox, positions[k, 1], oz)
            pc = Poly3DCollection(
                faces, alpha=mesh_alpha, facecolors=self.body_color,
                edgecolors="#333333", linewidths=edge_lw)
            ax.add_collection3d(pc)
            self._mpl_collections.append(pc)

    def update_mpl_bodies(self, positions, orientations, alive,
                          results, grid_ij, colors):
        """Update all matplotlib mesh collections in-place."""
        for k in range(self.N):
            ox, oz = self.offsets[k]
            faces = _transform_mesh(
                self._mpl_ref_mesh,
                orientations[k, 0], orientations[k, 1],
                orientations[k, 2], orientations[k, 3],
                ox, positions[k, 1], oz)
            self._mpl_collections[k].set_verts(faces)
            if not alive[k]:
                i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
                val = results[i2, j2]
                if not np.isnan(val):
                    c = colors.get(int(val), "#999999")
                    self._mpl_collections[k].set_facecolors(c)
