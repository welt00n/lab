"""
GPU-accelerated 3D scene for rigid-body drop experiments.

Wraps PyVista (VTK/OpenGL) to render N falling bodies on a floor.
Each body is a separate VTK actor sharing a template mesh, with a
per-frame ``user_matrix`` (4x4 affine) built from the physics
quaternion + position.  All polygon rendering runs on the GPU via
OpenGL — no CPU depth-sorting like matplotlib's Poly3DCollection.

Used by
-------
* ``lab/experiments/live_dashboard.py``  — live physics
* ``lab/experiments/drop_experiment.py`` — batch replay with time slider
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Visual mesh dimensions — decoupled from tiny real-world physics sizes
VIS_COIN_RADIUS = 0.15
VIS_COIN_HT = 0.01
VIS_CUBE_HALF = 0.15


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_template(shape: str):
    """Centred-at-origin template mesh in *display* space (z-up)."""
    if shape == "coin":
        return pv.Cylinder(
            radius=VIS_COIN_RADIUS,
            height=2 * VIS_COIN_HT,
            direction=(0, 0, 1),
            center=(0, 0, 0),
            resolution=20,
            capping=True,
        )
    h = VIS_CUBE_HALF
    return pv.Box(bounds=(-h, h, -h, h, -h, h))


def build_user_matrix(qw, qx, qy, qz, px, py, pz, ox=0.0, oz=0.0):
    """
    Physics state → VTK 4×4 user_matrix.

    Physics convention : y-up  (pos[1] = height).
    Display convention : z-up.
    Mapping: display = (phys_x + ox,  phys_z + oz,  phys_y).

    The rotation is conjugated by the y↔z swap matrix S so that body
    orientations display correctly.
    """
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

    # S @ R @ S  where S swaps rows/cols 1 and 2
    return np.array(
        [
            [r00, r02, r01, px + ox],
            [r20, r22, r21, pz + oz],
            [r10, r12, r11, py],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_grid_offsets(nh, na, grid_ij, shape="coin"):
    """
    Compute display-space ``(x, y)`` offsets for an ``nh × na`` grid.

    Returns an ``(N, 2)`` array where ``N = nh * na``.
    """
    N = len(grid_ij)
    obj_diam = 0.30
    gap = max(0.08, 0.40 / max(1, max(nh, na) ** 0.5))
    spacing = obj_diam + gap
    offsets = np.empty((N, 2), dtype=np.float64)
    for k in range(N):
        i_row = int(grid_ij[k, 0])
        j_col = int(grid_ij[k, 1])
        offsets[k, 0] = (j_col - (na - 1) / 2.0) * spacing
        offsets[k, 1] = (i_row - (nh - 1) / 2.0) * spacing
    return offsets


# ------------------------------------------------------------------
# DropScene
# ------------------------------------------------------------------

class DropScene:
    """
    PyVista / VTK 3-D scene for *N* dropping rigid bodies.

    Parameters
    ----------
    shape : ``"coin"`` or ``"cube"``
    N : int
        Number of bodies.
    offsets : ndarray, shape ``(N, 2)``
        Display-space ``(x, y)`` grid offset per body (z = height).
    max_height : float
        Maximum drop height (for camera framing).
    title : str
        Window title text.
    """

    def __init__(
        self,
        shape: str,
        N: int,
        offsets: np.ndarray,
        max_height: float,
        title: str = "drop scene",
    ):
        if not HAS_PYVISTA:
            raise ImportError(
                "pyvista required for GPU-accelerated 3-D rendering. "
                "Install with:  pip install pyvista"
            )

        self.shape = shape
        self.N = N
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.max_height = max_height
        self._settled: set[int] = set()
        self._default_rgb = (0.47, 0.60, 0.73) if shape == "coin" else (0.73, 0.60, 0.47)

        alpha = 0.78 if N > 100 else 0.88
        template = _make_template(shape)

        self.plotter = pv.Plotter(title=title, window_size=(900, 700))
        self.plotter.set_background("white")
        try:
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass

        # Floor plane
        xr = self.offsets[:, 0]
        yr = self.offsets[:, 1]
        pad = 0.5
        floor = pv.Plane(
            center=((xr.min() + xr.max()) / 2, (yr.min() + yr.max()) / 2, 0),
            direction=(0, 0, 1),
            i_size=float(xr.max() - xr.min()) + 2 * pad,
            j_size=float(yr.max() - yr.min()) + 2 * pad,
        )
        self.plotter.add_mesh(floor, color="#cccccc", opacity=0.35)

        # One VTK actor per body (shared template geometry)
        self.actors = []
        for _ in range(N):
            actor = self.plotter.add_mesh(
                template.copy(),
                color=self._default_rgb,
                opacity=alpha,
                smooth_shading=True,
            )
            self.actors.append(actor)

        # Camera
        cx = (xr.min() + xr.max()) / 2
        cy = (yr.min() + yr.max()) / 2
        span = max(float(xr.max() - xr.min()), float(yr.max() - yr.min()), 1.0)
        dist = span * 1.5 + max_height * 0.4
        self.plotter.camera_position = [
            (cx - dist * 0.65, cy - dist * 0.65, max_height * 0.55),
            (cx, cy, max_height * 0.20),
            (0, 0, 1),
        ]

        self._title_text = title
        self._title_actor = self.plotter.add_text(
            title, position="upper_left", font_size=10, color="black"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_nonblocking(self):
        """Open the VTK window and return immediately."""
        self.plotter.show(interactive_update=True)

    def update_all(self, positions, orientations, alive=None):
        """
        Set ``user_matrix`` for every body that has not yet settled.

        Parameters
        ----------
        positions : ``(N, 3)`` — physics ``[x, y_height, z]``
        orientations : ``(N, 4)`` — quaternion ``[w, x, y, z]``
        alive : ``(N,)`` bool, optional
        """
        for k in range(self.N):
            if k in self._settled:
                continue
            qw, qx, qy, qz = orientations[k]
            px, py, pz = positions[k]
            ox, oz = self.offsets[k]
            self.actors[k].user_matrix = build_user_matrix(
                qw, qx, qy, qz, px, py, pz, ox, oz
            )

    def mark_settled(self, k: int, color=None):
        """Freeze body *k* at its current transform and set its outcome colour."""
        self._settled.add(k)
        if color is not None:
            from matplotlib.colors import to_rgb

            r, g, b = to_rgb(color)
            self.actors[k].GetProperty().SetColor(r, g, b)

    def reset(self):
        """Clear the settled set and restore default colours (for rewind)."""
        self._settled.clear()
        r, g, b = self._default_rgb
        for actor in self.actors:
            actor.GetProperty().SetColor(r, g, b)

    def set_title(self, text: str):
        if text != self._title_text:
            self.plotter.remove_actor(self._title_actor)
            self._title_actor = self.plotter.add_text(
                text, position="upper_left", font_size=10, color="black"
            )
            self._title_text = text

    def render(self):
        """Process VTK events and push one frame to the GPU."""
        try:
            iren = getattr(self.plotter, "iren", None)
            if iren is not None:
                iren.process_events()
            self.plotter.render()
        except Exception:
            pass

    def close(self):
        try:
            self.plotter.close()
        except Exception:
            pass
