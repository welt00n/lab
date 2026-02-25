"""
Drop experiment — sweep height x tilt-angle for rigid bodies and
classify which face lands on the floor.

Supports coin, cube, and rod.  Simulations run in parallel across
CPU cores via ProcessPoolExecutor, with an optional per-result
callback for progressive live plotting.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from lab.core import quaternion as quat
from lab.systems.rigid_body.objects import RigidBody
from lab.systems.rigid_body.fields import GravityField
from lab.systems.rigid_body.constraints import FloorConstraint
from lab.systems.rigid_body.world import World


# ------------------------------------------------------------------
# Outcome classification
# ------------------------------------------------------------------

_BODY_AXES = np.array([
    [1, 0, 0], [-1, 0, 0],   # +x, -x
    [0, 1, 0], [0, -1, 0],   # +y, -y
    [0, 0, 1], [0, 0, -1],   # +z, -z
], dtype=float)

CUBE_FACE_LABELS = ["+x", "-x", "+y", "-y", "+z", "-z"]


def classify_coin(orientation):
    """Heads (+1), Tails (-1), or Edge (0)."""
    world_y = quat.rotate_vector(orientation, np.array([0.0, 1.0, 0.0]))
    y = world_y[1]
    if y > 0.1:
        return 1
    if y < -0.1:
        return -1
    return 0


def classify_cube_face(orientation):
    """Return 0-5 for which body-frame axis points most downward."""
    R = quat.to_rotation_matrix(orientation)
    world_axes = R @ _BODY_AXES.T          # (3, 6)
    y_components = world_axes[1, :]        # world-y of each axis
    return int(np.argmin(y_components))


def classify_rod_end(orientation):
    """+1 if body +x end is lower, -1 if -x end is lower, 0 if flat."""
    world_x = quat.rotate_vector(orientation, np.array([1.0, 0.0, 0.0]))
    y = world_x[1]
    if y < -0.1:
        return 1
    if y > 0.1:
        return -1
    return 0


def classify(shape, orientation):
    """Dispatch to the appropriate classifier."""
    if shape == "coin":
        return classify_coin(orientation)
    if shape == "cube":
        return classify_cube_face(orientation)
    if shape == "rod":
        return classify_rod_end(orientation)
    raise ValueError(f"Unknown shape: {shape!r}")


# ------------------------------------------------------------------
# Single drop simulation
# ------------------------------------------------------------------

_AXIS_MAP = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

_BUILDERS = {
    "coin": lambda **kw: RigidBody.coin(mass=1.0, radius=0.15, **kw),
    "cube": lambda **kw: RigidBody.cube(mass=1.0, side=0.3, **kw),
    "rod":  lambda **kw: RigidBody.rod(mass=1.0, length=1.0, **kw),
}

_SETTLE_HEIGHTS = {"coin": 0.20, "cube": 0.35, "rod": 0.55}


def drop_body(shape, height, tilt_axis, tilt_angle,
              dt=0.001, restitution=0.6, friction=0.5,
              rolling_resistance=0.05, g=9.81, duration=None):
    """
    Drop a rigid body from *height* with initial tilt and return the
    final orientation quaternion after settling.
    """
    axis_vec = _AXIS_MAP[tilt_axis]
    orientation = quat.from_axis_angle(axis_vec, tilt_angle)

    body = _BUILDERS[shape](
        position=np.array([0.0, height, 0.0]),
        momentum=np.zeros(3),
        orientation=orientation,
    )

    world = World()
    world.add_particle(body)
    world.add_field(GravityField(g))
    world.add_constraint(FloorConstraint(
        restitution, friction=friction,
        rolling_resistance=rolling_resistance))

    if duration is None:
        t_fall = np.sqrt(2 * height / g)
        duration = max(t_fall * 8, 2.0)

    nsteps = int(np.ceil(duration / dt))
    settle_h = _SETTLE_HEIGHTS.get(shape, 0.5)
    settled_count = 0

    for _ in range(nsteps):
        world.step(dt)

        if body.kinetic_energy() < 1e-6 and body.position[1] < settle_h:
            settled_count += 1
            if settled_count > 200:
                break
        else:
            settled_count = 0

    return body.orientation.copy()


# ------------------------------------------------------------------
# Worker (module-level for pickling)
# ------------------------------------------------------------------

def _worker(args):
    """Run one drop and classify.  Returns (i, j, result)."""
    i, j, shape, height, tilt_axis, tilt_angle, sim_params = args
    orientation = drop_body(shape, height, tilt_axis, tilt_angle, **sim_params)
    return i, j, classify(shape, orientation)


# ------------------------------------------------------------------
# Parallel sweep
# ------------------------------------------------------------------

def sweep_drop(shape, heights, angles, tilt_axis="x",
               workers=None, callback=None, **sim_params):
    """
    Run a 2D grid of drops: height x tilt-angle.

    Parameters
    ----------
    shape : str
        "coin", "cube", or "rod".
    heights : array-like
        Drop heights to sweep.
    angles : array-like
        Tilt angles to sweep (radians).
    tilt_axis : str
        "x", "y", or "z".
    workers : int or None
        Number of parallel workers.  None = os.cpu_count().
        Use 1 for sequential execution.
    callback : callable or None
        Called after each result: callback(i, j, result, done, total).
    **sim_params
        Forwarded to drop_body (dt, restitution, friction, etc.).

    Returns
    -------
    results : ndarray, shape (len(heights), len(angles))
    """
    heights = np.asarray(heights)
    angles = np.asarray(angles)
    nh, na = len(heights), len(angles)
    total = nh * na
    results = np.full((nh, na), np.nan)

    jobs = []
    for i, h in enumerate(heights):
        for j, a in enumerate(angles):
            jobs.append((i, j, shape, float(h), tilt_axis, float(a), sim_params))

    if workers is None:
        workers = os.cpu_count() or 4

    done = 0

    if workers == 1:
        for job in jobs:
            i, j, result = _worker(job)
            results[i, j] = result
            done += 1
            if callback:
                callback(i, j, result, done, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, job): job for job in jobs}
            for future in as_completed(futures):
                i, j, result = future.result()
                results[i, j] = result
                done += 1
                if callback:
                    callback(i, j, result, done, total)

    return results.astype(int)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

_COIN_COLORS = {1: "#1f77b4", -1: "#d62728", 0: "#cccccc"}
_COIN_LABELS = {1: "Heads", -1: "Tails", 0: "Edge"}

_CUBE_COLORS = {
    0: "#1f77b4", 1: "#aec7e8",   # +x blue,   -x light blue
    2: "#d62728", 3: "#ff9896",   # +y red,    -y light red
    4: "#2ca02c", 5: "#98df8a",   # +z green,  -z light green
}
_CUBE_LABELS = dict(enumerate(CUBE_FACE_LABELS))

_ROD_COLORS = {1: "#1f77b4", -1: "#d62728", 0: "#cccccc"}
_ROD_LABELS = {1: "+x end down", -1: "-x end down", 0: "flat"}

_SHAPE_PALETTE = {
    "coin": (_COIN_COLORS, _COIN_LABELS),
    "cube": (_CUBE_COLORS, _CUBE_LABELS),
    "rod":  (_ROD_COLORS,  _ROD_LABELS),
}


def _results_to_rgb(results, colors, nan_color=(0.85, 0.85, 0.85)):
    """Convert integer result grid to an RGB image array."""
    nh, na = results.shape
    rgb = np.full((nh, na, 3), nan_color)
    import matplotlib.colors as mcolors
    for val, hex_color in colors.items():
        r, g, b = mcolors.to_rgb(hex_color)
        mask = results == val
        rgb[mask] = [r, g, b]
    return rgb


def plot_drop_map(heights, angles, results, shape, tilt_axis="x", ax=None):
    """
    Plot the outcome map as a color image.

    Returns (fig, ax, img) so that the caller can update img.set_data()
    for progressive rendering.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors, labels = _SHAPE_PALETTE[shape]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    rgb = _results_to_rgb(results, colors)
    extent = [
        np.degrees(angles[0]), np.degrees(angles[-1]),
        heights[0], heights[-1],
    ]
    img = ax.imshow(rgb, origin="lower", aspect="auto", extent=extent,
                    interpolation="nearest")

    ax.set_xlabel(f"initial tilt about {tilt_axis}-axis (degrees)")
    ax.set_ylabel("drop height (m)")
    ax.set_title(f"{shape} drop outcome map")

    legend_elements = [
        Patch(facecolor=c, edgecolor="gray", label=labels[v])
        for v, c in colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    return fig, ax, img
