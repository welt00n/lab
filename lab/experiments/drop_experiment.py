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


# ------------------------------------------------------------------
# Animated result viewer with playback controls
# ------------------------------------------------------------------

def animate_drop_results(heights, angles, results, shape,
                         tilt_axis="x", title_suffix=""):
    """
    Time-based animated reveal of a pre-computed outcome map with a
    3D physical scene showing all bodies falling in real time.

    Uses PyVista for GPU-accelerated 3D rendering when available,
    falling back to matplotlib otherwise.

    Parameters
    ----------
    heights, angles : array-like
    results : ndarray, shape (nh, na) — integer outcomes
    shape : str — "coin" or "cube"
    tilt_axis : str
    title_suffix : str — appended to the figure title
    """
    try:
        from lab.visualization.pyvista_scene import DropScene, HAS_PYVISTA
    except ImportError:
        HAS_PYVISTA = False

    if HAS_PYVISTA:
        return _animate_pyvista(heights, angles, results, shape,
                                tilt_axis, title_suffix)
    return _animate_mpl_fallback(heights, angles, results, shape,
                                 tilt_axis, title_suffix)


def _animate_pyvista(heights, angles, results, shape,
                     tilt_axis="x", title_suffix=""):
    """PyVista 3D scene + matplotlib 2D panels for batch result replay."""
    import time as _time
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from lab.visualization.pyvista_scene import DropScene

    from lab.experiments.live_dashboard import (
        _step_all, _classify_jit, _qaa,
        _SHAPE_TO_ID, COIN_RADIUS, CUBE_HALF_SIDE,
    )

    colors, labels = _SHAPE_PALETTE[shape]
    nh, na = results.shape
    g = 9.81
    shape_id = _SHAPE_TO_ID[shape]

    settle_times = np.sqrt(2.0 * np.maximum(heights, 0.01) / g) * 6.0
    t_max = float(settle_times[-1])
    n_frames = max(nh, 200)
    time_axis = np.linspace(0, t_max, n_frames)
    full_rgb = _results_to_rgb(results, colors)
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    N = nh * na
    body_grid = [(hi, ai) for hi in range(nh) for ai in range(na)]
    ax_f, ay_f, az_f = {"x": (1., 0., 0.),
                        "y": (0., 1., 0.),
                        "z": (0., 0., 1.)}[tilt_axis]

    s_pos = np.zeros((N, 3), dtype=np.float64)
    s_mom = np.zeros((N, 3), dtype=np.float64)
    s_ori = np.zeros((N, 4), dtype=np.float64)
    s_amom = np.zeros((N, 3), dtype=np.float64)
    s_alive = np.ones(N, dtype=np.bool_)
    s_sc = np.zeros(N, dtype=np.int64)
    s_alive_idx = np.arange(N, dtype=np.int64)
    s_n_alive = [N]

    for si, (hi, ai) in enumerate(body_grid):
        s_pos[si] = [0.0, float(heights[hi]), 0.0]
        w, x, y, z = _qaa(ax_f, ay_f, az_f, float(angles[ai]))
        s_ori[si] = [w, x, y, z]

    settle_h = {"coin": 5.0 * COIN_RADIUS,
                "cube": 3.0 * CUBE_HALF_SIDE}.get(shape, 0.05)

    # Grid offsets
    obj_diam = 0.30
    gap = max(0.08, 0.40 / max(1, max(nh, na) ** 0.5))
    spacing = obj_diam + gap
    scene_offsets = np.empty((N, 2), dtype=np.float64)
    for si, (hi, ai) in enumerate(body_grid):
        scene_offsets[si, 0] = (ai - (na - 1) / 2) * spacing
        scene_offsets[si, 1] = (hi - (nh - 1) / 2) * spacing
    max_h = float(heights.max())

    # JIT warm-up
    print("  Compiling 3D physics...", end=" ", flush=True)
    _w = np.zeros((1, 3))
    _wo = np.array([[1., 0., 0., 0.]])
    _wa = np.ones(1, dtype=np.bool_)
    _wi = np.zeros(1, dtype=np.int64)
    _wsc = np.zeros(1, dtype=np.int64)
    _step_all(_w.copy(), _w.copy(), _wo.copy(), _w.copy(),
              _wa.copy(), _wsc.copy(), _wi.copy(), 1,
              shape_id, 0.0005, 9.81, 0.6, 0.5, 0.05, 0.5, 1)
    print("done.", flush=True)

    # PyVista 3D scene
    scene = DropScene(shape, N, scene_offsets, max_h,
                      title=f"{shape} drop \u2014 result viewer")
    scene.show_nonblocking()

    # Matplotlib 2D panels + controls
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 0.6],
                          height_ratios=[1, 0.06],
                          hspace=0.30, wspace=0.22,
                          left=0.06, right=0.97, top=0.93, bottom=0.10)

    ax_map = fig.add_subplot(gs[0, 0])
    img = ax_map.imshow(full_rgb, origin="lower", aspect="auto",
                        extent=extent, interpolation="nearest")
    ax_map.set_xlabel(f"tilt about {tilt_axis}-axis (deg)")
    ax_map.set_ylabel("height (m)")
    ax_map.set_title(f"{shape} outcome map{title_suffix}")
    legend_elems = [Patch(facecolor=c, edgecolor="gray", label=labels[v])
                    for v, c in colors.items()]
    ax_map.legend(handles=legend_elems, loc="upper right", fontsize=7)

    ax_hist = fig.add_subplot(gs[0, 1])
    outcome_keys = sorted(colors.keys())
    bar_colors_list = [colors[k] for k in outcome_keys]
    bar_labels_list = [labels[k] for k in outcome_keys]
    bar_counts = [int(np.sum(results == k)) for k in outcome_keys]
    bars = ax_hist.bar(range(len(outcome_keys)), bar_counts,
                       color=bar_colors_list)
    ax_hist.set_xticks(range(len(outcome_keys)))
    ax_hist.set_xticklabels(bar_labels_list, fontsize=7, rotation=30)
    ax_hist.set_ylabel("count")
    ax_hist.set_title("distribution")
    ax_hist.set_ylim(0, max(1, max(bar_counts)) * 1.15)

    ax_slider = fig.add_subplot(gs[1, 0])
    slider = Slider(ax_slider, "time (s)", 0, n_frames - 1,
                    valinit=0, valstep=1, valfmt="%d")
    slider.valtext.set_text("t = 0.00 s")

    ax_play = fig.add_axes([0.72, 0.02, 0.06, 0.04])
    ax_pbtn = fig.add_axes([0.79, 0.02, 0.06, 0.04])
    ax_fwd = fig.add_axes([0.86, 0.02, 0.06, 0.04])
    ax_back = fig.add_axes([0.65, 0.02, 0.06, 0.04])
    btn_play = Button(ax_play, "> Play")
    btn_pause = Button(ax_pbtn, "|| Pause")
    btn_fwd = Button(ax_fwd, ">> Step")
    btn_back = Button(ax_back, "<< Back")

    dt_phys = 0.0005
    dt_frame = t_max / n_frames
    phys_steps_per_frame = max(1, int(dt_frame / dt_phys))

    state = {"frame": 0, "playing": False, "updating": False,
             "phys_frame": 0,
             "play_wall_t0": 0.0, "play_sim_t0": 0.0}

    def _reset_physics():
        for si, (hi, ai) in enumerate(body_grid):
            s_pos[si] = [0.0, float(heights[hi]), 0.0]
            s_mom[si] = [0.0, 0.0, 0.0]
            w, x, y, z = _qaa(ax_f, ay_f, az_f, float(angles[ai]))
            s_ori[si] = [w, x, y, z]
            s_amom[si] = [0.0, 0.0, 0.0]
            s_alive[si] = True
            s_sc[si] = 0
        s_alive_idx[:N] = np.arange(N)
        s_n_alive[0] = N
        scene.reset()

    def _step_physics_to(target_frame):
        current = state["phys_frame"]
        if target_frame < current:
            _reset_physics()
            current = 0
        for f in range(current, target_frame):
            n_al = s_n_alive[0]
            if n_al == 0:
                break
            _, new_n = _step_all(
                s_pos, s_mom, s_ori, s_amom, s_alive, s_sc,
                s_alive_idx, n_al,
                shape_id, dt_phys, g,
                0.6, 0.5, 0.05, settle_h, phys_steps_per_frame)
            s_n_alive[0] = new_n
        state["phys_frame"] = target_frame

    def _draw_frame(f):
        if state["updating"]:
            return
        state["updating"] = True
        f = max(0, min(f, n_frames - 1))
        state["frame"] = f
        t = time_axis[f]

        _step_physics_to(f)

        # Mark settled bodies in the 3D scene
        for si in range(N):
            if not s_alive[si]:
                hi, ai = body_grid[si]
                val = results[hi, ai]
                c = colors.get(int(val), "#999999")
                scene.mark_settled(si, c)

        try:
            scene.update_all(s_pos, s_ori, s_alive)
            n_al = int(s_alive.sum())
            scene.set_title(
                f"{shape}s \u2014 t = {t:.1f} s \u2014 {n_al} falling")
            scene.render()
        except Exception:
            pass

        slider.set_val(f)
        slider.valtext.set_text(f"t = {t:.2f} s")
        fig.canvas.draw_idle()
        state["updating"] = False

    def on_slider(val):
        _draw_frame(int(val))

    def on_play(_):
        state["playing"] = True
        state["play_wall_t0"] = _time.monotonic()
        state["play_sim_t0"] = time_axis[state["frame"]]

    def on_pause(_):
        state["playing"] = False

    def on_fwd(_):
        state["playing"] = False
        _draw_frame(state["frame"] + 1)

    def on_back(_):
        state["playing"] = False
        _draw_frame(state["frame"] - 1)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_fwd.on_clicked(on_fwd)
    btn_back.on_clicked(on_back)

    def _update(_anim_frame):
        if not state["playing"]:
            return
        if state["frame"] >= n_frames - 1:
            state["playing"] = False
            return
        elapsed = _time.monotonic() - state["play_wall_t0"]
        target_t = state["play_sim_t0"] + elapsed
        target_f = int(np.searchsorted(time_axis, target_t, side="right")) - 1
        target_f = max(state["frame"], min(target_f, n_frames - 1))
        if target_f != state["frame"]:
            _draw_frame(target_f)

    ani = animation.FuncAnimation(fig, _update, interval=33,
                                  cache_frame_data=False)
    _draw_frame(0)
    plt.show()
    scene.close()
    return ani


def _animate_mpl_fallback(heights, angles, results, shape,
                          tilt_axis="x", title_suffix=""):
    """Matplotlib-only fallback for animate_drop_results."""
    import time as _time
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from lab.experiments.live_dashboard import (
        _coin_mesh, _cube_mesh, _transform_mesh,
        _step_all, _classify_jit, _qaa,
        _SHAPE_TO_ID, COIN_RADIUS, CUBE_HALF_SIDE,
    )

    colors, labels = _SHAPE_PALETTE[shape]
    nh, na = results.shape
    g = 9.81
    shape_id = _SHAPE_TO_ID[shape]

    settle_times = np.sqrt(2.0 * np.maximum(heights, 0.01) / g) * 6.0
    t_max = float(settle_times[-1])

    n_frames = max(nh, 200)
    time_axis = np.linspace(0, t_max, n_frames)

    full_rgb = _results_to_rgb(results, colors)
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    N_total = nh * na
    ns = N_total
    body_grid = []
    for hi in range(nh):
        for ai in range(na):
            body_grid.append((hi, ai))

    ax_f, ay_f, az_f = {"x": (1., 0., 0.),
                        "y": (0., 1., 0.),
                        "z": (0., 0., 1.)}[tilt_axis]

    s_pos = np.zeros((ns, 3), dtype=np.float64)
    s_mom = np.zeros((ns, 3), dtype=np.float64)
    s_ori = np.zeros((ns, 4), dtype=np.float64)
    s_amom = np.zeros((ns, 3), dtype=np.float64)
    s_alive = np.ones(ns, dtype=np.bool_)
    s_sc = np.zeros(ns, dtype=np.int64)
    s_alive_idx = np.arange(ns, dtype=np.int64)
    s_n_alive = [ns]

    for si, (hi, ai) in enumerate(body_grid):
        s_pos[si] = [0.0, float(heights[hi]), 0.0]
        w, x, y, z = _qaa(ax_f, ay_f, az_f, float(angles[ai]))
        s_ori[si] = [w, x, y, z]

    settle_h = {"coin": 5.0 * COIN_RADIUS,
                "cube": 3.0 * CUBE_HALF_SIDE,
                "rod": 0.55}.get(shape, 0.05)
    ref_mesh = _coin_mesh() if shape == "coin" else _cube_mesh()

    obj_diam = {"coin": 0.30, "cube": 0.30}.get(shape, 0.30)
    gap = max(0.08, 0.40 / max(1, max(nh, na) ** 0.5))
    scene_spacing = obj_diam + gap
    scene_offsets = []
    for si, (hi, ai) in enumerate(body_grid):
        ox = (ai - (na - 1) / 2) * scene_spacing
        oz = (hi - (nh - 1) / 2) * scene_spacing
        scene_offsets.append((ox, oz))

    print("  Compiling 3D physics...", end=" ", flush=True)
    _w = np.zeros((1, 3))
    _wo = np.array([[1., 0., 0., 0.]])
    _wa = np.ones(1, dtype=np.bool_)
    _wi = np.zeros(1, dtype=np.int64)
    _wsc = np.zeros(1, dtype=np.int64)
    _step_all(_w.copy(), _w.copy(), _wo.copy(), _w.copy(),
              _wa.copy(), _wsc.copy(), _wi.copy(), 1,
              shape_id, 0.0005, 9.81, 0.6, 0.5, 0.05, 0.5, 1)
    print("done.", flush=True)

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 0.6],
                          height_ratios=[1, 0.06],
                          hspace=0.30, wspace=0.22,
                          left=0.04, right=0.97, top=0.93, bottom=0.10)

    ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
    max_h = float(heights.max())
    scene_w = na * scene_spacing / 2 + 0.3
    scene_d = nh * scene_spacing / 2 + 0.3
    ax_scene.set_xlim(-scene_w, scene_w)
    ax_scene.set_ylim(-scene_d, scene_d)
    ax_scene.set_zlim(-0.05, max_h * 1.1)
    ax_scene.set_xlabel("x", fontsize=7)
    ax_scene.set_ylabel("z", fontsize=7)
    ax_scene.set_zlabel("height (m)", fontsize=8)
    ax_scene.set_title(f"{shape}s \u2014 physical view")
    ax_scene.view_init(elev=18, azim=-60)
    ax_scene.xaxis.set_tick_params(labelsize=6)
    ax_scene.yaxis.set_tick_params(labelsize=6)
    ax_scene.zaxis.set_tick_params(labelsize=7)

    fx = np.array([-scene_w, scene_w, scene_w, -scene_w])
    fz = np.array([-scene_d, -scene_d, scene_d, scene_d])
    fy = np.zeros(4)
    floor_poly = Poly3DCollection(
        [list(zip(fx, fz, fy))],
        alpha=0.15, facecolors="#888888", edgecolors="#aaaaaa", linewidths=0.5)
    ax_scene.add_collection3d(floor_poly)

    face_color = "#7799bb" if shape == "coin" else "#bb9977"
    edge_lw = 0.3 if ns > 50 else 0.4
    mesh_alpha = 0.75 if ns > 100 else 0.85
    mesh_collections = []
    for si in range(ns):
        ox, oz = scene_offsets[si]
        faces = _transform_mesh(ref_mesh,
                                s_ori[si, 0], s_ori[si, 1],
                                s_ori[si, 2], s_ori[si, 3],
                                ox, s_pos[si, 1], oz)
        pc = Poly3DCollection(faces, alpha=mesh_alpha, facecolors=face_color,
                              edgecolors="#333333", linewidths=edge_lw)
        ax_scene.add_collection3d(pc)
        mesh_collections.append(pc)

    ax_map = fig.add_subplot(gs[0, 1])
    img = ax_map.imshow(full_rgb, origin="lower", aspect="auto",
                        extent=extent, interpolation="nearest")
    ax_map.set_xlabel(f"tilt about {tilt_axis}-axis (deg)")
    ax_map.set_ylabel("height (m)")
    ax_map.set_title(f"{shape} outcome map{title_suffix}")
    legend_elems = [Patch(facecolor=c, edgecolor="gray", label=labels[v])
                    for v, c in colors.items()]
    ax_map.legend(handles=legend_elems, loc="upper right", fontsize=7)

    ax_hist = fig.add_subplot(gs[0, 2])
    outcome_keys = sorted(colors.keys())
    bar_colors = [colors[k] for k in outcome_keys]
    bar_labels_list = [labels[k] for k in outcome_keys]
    bar_counts = [int(np.sum(results == k)) for k in outcome_keys]
    bars = ax_hist.bar(range(len(outcome_keys)), bar_counts,
                       color=bar_colors)
    ax_hist.set_xticks(range(len(outcome_keys)))
    ax_hist.set_xticklabels(bar_labels_list, fontsize=7, rotation=30)
    ax_hist.set_ylabel("count")
    ax_hist.set_title("distribution")
    ax_hist.set_ylim(0, max(1, max(bar_counts)) * 1.15)

    ax_slider = fig.add_subplot(gs[1, :2])
    slider = Slider(ax_slider, "time (s)", 0, n_frames - 1,
                    valinit=0, valstep=1, valfmt="%d")
    slider.valtext.set_text("t = 0.00 s")

    ax_play = fig.add_axes([0.72, 0.02, 0.06, 0.04])
    ax_pause = fig.add_axes([0.79, 0.02, 0.06, 0.04])
    ax_fwd = fig.add_axes([0.86, 0.02, 0.06, 0.04])
    ax_back = fig.add_axes([0.65, 0.02, 0.06, 0.04])
    btn_play = Button(ax_play, "> Play")
    btn_pause = Button(ax_pause, "|| Pause")
    btn_fwd = Button(ax_fwd, ">> Step")
    btn_back = Button(ax_back, "<< Back")

    dt_phys = 0.0005
    dt_frame = t_max / n_frames
    phys_steps_per_frame = max(1, int(dt_frame / dt_phys))

    state = {"frame": 0, "playing": False, "updating": False,
             "phys_frame": 0,
             "play_wall_t0": 0.0, "play_sim_t0": 0.0}

    def _step_physics_to(target_frame):
        current = state["phys_frame"]
        if target_frame < current:
            for si, (hi, ai) in enumerate(body_grid):
                s_pos[si] = [0.0, float(heights[hi]), 0.0]
                s_mom[si] = [0.0, 0.0, 0.0]
                w, x, y, z = _qaa(ax_f, ay_f, az_f, float(angles[ai]))
                s_ori[si] = [w, x, y, z]
                s_amom[si] = [0.0, 0.0, 0.0]
                s_alive[si] = True
                s_sc[si] = 0
            s_alive_idx[:ns] = np.arange(ns)
            s_n_alive[0] = ns
            current = 0
        for f in range(current, target_frame):
            n_al = s_n_alive[0]
            if n_al == 0:
                break
            _, new_n = _step_all(
                s_pos, s_mom, s_ori, s_amom, s_alive, s_sc,
                s_alive_idx, n_al,
                shape_id, dt_phys, g,
                0.6, 0.5, 0.05, settle_h, phys_steps_per_frame)
            s_n_alive[0] = new_n
        state["phys_frame"] = target_frame

    def _draw_frame(f):
        if state["updating"]:
            return
        state["updating"] = True
        f = max(0, min(f, n_frames - 1))
        state["frame"] = f
        t = time_axis[f]
        _step_physics_to(f)
        for si in range(ns):
            ox, oz = scene_offsets[si]
            faces = _transform_mesh(
                ref_mesh,
                s_ori[si, 0], s_ori[si, 1],
                s_ori[si, 2], s_ori[si, 3],
                ox, s_pos[si, 1], oz)
            mesh_collections[si].set_verts(faces)
            if not s_alive[si]:
                hi, ai = body_grid[si]
                val = results[hi, ai]
                c = colors.get(int(val), "#999999")
                mesh_collections[si].set_facecolors(c)
        n_scene_alive = int(s_alive.sum())
        ax_scene.set_title(
            f"{shape}s \u2014 t = {t:.1f} s \u2014 "
            f"{n_scene_alive} falling")
        slider.set_val(f)
        slider.valtext.set_text(f"t = {t:.2f} s")
        fig.canvas.draw_idle()
        state["updating"] = False

    def on_slider(val):
        _draw_frame(int(val))

    def on_play(_):
        state["playing"] = True
        state["play_wall_t0"] = _time.monotonic()
        state["play_sim_t0"] = time_axis[state["frame"]]

    def on_pause(_):
        state["playing"] = False

    def on_fwd(_):
        state["playing"] = False
        _draw_frame(state["frame"] + 1)

    def on_back(_):
        state["playing"] = False
        _draw_frame(state["frame"] - 1)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_fwd.on_clicked(on_fwd)
    btn_back.on_clicked(on_back)

    def _update(_anim_frame):
        if not state["playing"]:
            return
        if state["frame"] >= n_frames - 1:
            state["playing"] = False
            return
        elapsed = _time.monotonic() - state["play_wall_t0"]
        target_t = state["play_sim_t0"] + elapsed
        target_f = int(np.searchsorted(time_axis, target_t, side="right")) - 1
        target_f = max(state["frame"], min(target_f, n_frames - 1))
        if target_f != state["frame"]:
            _draw_frame(target_f)

    ani = animation.FuncAnimation(fig, _update, interval=33,
                                  cache_frame_data=False)
    _draw_frame(0)
    plt.show()
    return ani
