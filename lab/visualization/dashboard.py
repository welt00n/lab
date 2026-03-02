"""
Dashboard compositor — wires visualization primitives + experiment
config + physics into live, replay, and static modes.

Contains NO physics code and NO chart-drawing code — delegates to
rigid_body_jit for physics and to the visualization primitives
(sweep_grid, category_histogram, body_scene, playback_controls)
for all rendering.
"""

from __future__ import annotations

import time as _time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from lab.core.rigid_body_jit import (
    step_bodies, classify, warmup, quat_from_axis_angle,
)
from lab.visualization import sweep_grid, category_histogram
from lab.visualization.body_scene import BodyScene, HAS_PYVISTA
from lab.visualization.playback_controls import PlaybackControls

_AXIS_MAP = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _compute_offsets(nh, na, grid_ij):
    """Grid offsets for 3D scene layout."""
    N = len(grid_ij)
    obj_diam = 0.30
    gap = max(0.08, 0.40 / max(1, max(nh, na) ** 0.5))
    spacing = obj_diam + gap
    offsets = np.empty((N, 2), dtype=np.float64)
    for k in range(N):
        i_row, j_col = int(grid_ij[k, 0]), int(grid_ij[k, 1])
        offsets[k, 0] = (j_col - (na - 1) / 2) * spacing
        offsets[k, 1] = (i_row - (nh - 1) / 2) * spacing
    return offsets, spacing


def _init_state(experiment, heights, angles, tilt_axis):
    """Allocate and populate physics state arrays."""
    nh, na = len(heights), len(angles)
    N = nh * na
    ax_f, ay_f, az_f = _AXIS_MAP[tilt_axis]

    pos = np.zeros((N, 3), dtype=np.float64)
    mom = np.zeros((N, 3), dtype=np.float64)
    ori = np.zeros((N, 4), dtype=np.float64)
    amom = np.zeros((N, 3), dtype=np.float64)
    alive = np.ones(N, dtype=np.bool_)
    sc = np.zeros(N, dtype=np.int64)

    grid_ij = np.zeros((N, 2), dtype=np.int64)
    idx = 0
    for i, h in enumerate(heights):
        for j, a in enumerate(angles):
            pos[idx] = [0.0, float(h), 0.0]
            w, x, y, z = quat_from_axis_angle(ax_f, ay_f, az_f, float(a))
            ori[idx] = [w, x, y, z]
            grid_ij[idx] = [i, j]
            idx += 1

    return pos, mom, ori, amom, alive, sc, grid_ij


# ================================================================
# Live mode
# ================================================================

def run_live(experiment, heights, angles, tilt_axis="x"):
    """
    Live dashboard: physics + 3D scene + outcome map + histogram
    all updating per frame.
    """
    nh, na = len(heights), len(angles)
    N = nh * na
    shape_id = experiment.shape_id

    if N > 500:
        print(f"  NOTE: {N} objects is heavy for live mode. "
              f"Consider --nh 10 --na 15 for smoother animation.")

    print("  Compiling physics (first run only)...", end=" ", flush=True)
    warmup(shape_id)
    print("done.", flush=True)

    pos, mom, ori, amom, alive, sc, grid_ij = \
        _init_state(experiment, heights, angles, tilt_axis)
    offsets, spacing = _compute_offsets(nh, na, grid_ij)
    max_h = float(heights.max())

    results = np.full((nh, na), np.nan)
    outcome_keys = sorted(experiment.colors.keys())
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    use_pyvista = HAS_PYVISTA

    if use_pyvista:
        scene = BodyScene(
            experiment.mesh, N, offsets, max_h,
            body_color=experiment.body_color,
            title=f"{experiment.shape} drop \u2014 {N} simulations",
            use_pyvista=True)
        scene.show_nonblocking()

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.06],
                              hspace=0.30, wspace=0.22)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
    else:
        import warnings
        warnings.warn(
            "PyVista not installed \u2014 using matplotlib 3D (slower). "
            "Install with: pip install pyvista", stacklevel=2)

        scene = BodyScene(
            experiment.mesh, N, offsets, max_h,
            body_color=experiment.body_color,
            use_pyvista=False)

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1],
                              height_ratios=[1.2, 1, 0.01],
                              hspace=0.25, wspace=0.18)

        ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
        ref_mesh, edge_lw, mesh_alpha = scene.init_mpl_scene(
            ax_scene, spacing)
        ax_scene.set_title("physical view", fontsize=10)
        scene.create_mpl_bodies(ax_scene, pos, ori, edge_lw, mesh_alpha)

        ax_scatter = fig.add_subplot(gs[0, 1], projection="3d")
        _setup_scatter_axes(ax_scatter, nh, na, spacing, max_h,
                            experiment.shape, N)

        ax_map = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"{experiment.shape} drop \u2014 {N} simulations "
        f"({nh}h \u00d7 {na}a)",
        fontsize=13, fontweight="bold", y=0.97)

    img = sweep_grid.create(
        results, experiment.colors, extent,
        f"tilt about {tilt_axis}-axis (deg)", "height (m)",
        f"{experiment.shape} outcome map \u2014 0 / {N} settled",
        ax_map)
    sweep_grid.add_legend(ax_map, experiment.colors, experiment.labels)

    bars = category_histogram.create(
        outcome_keys, experiment.colors, experiment.labels, ax_hist)

    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.12,
                        top=0.90, wspace=0.22, hspace=0.30)

    paused = [False]
    speed_mult = [1.0]
    controls = PlaybackControls(
        fig,
        on_pause=lambda p: _set_flag(paused, p),
        on_speed=lambda v: _set_flag(speed_mult, v),
    )

    completed = [0]
    base_steps = max(10, min(150, 5000 // max(1, N)))
    all_done = [False]
    alive_idx = np.arange(N, dtype=np.int64)
    n_alive_box = [N]

    scatter_data = {}
    if not use_pyvista:
        scatter_data = _init_scatter(
            ax_scatter, N, grid_ij, nh, na, pos, spacing,
            experiment.colors)

    def update(_frame):
        if all_done[0] or paused[0]:
            return

        n_al = n_alive_box[0]
        steps = max(base_steps, base_steps * N // max(1, n_al))
        steps = int(min(steps * speed_mult[0], 5000))

        newly, new_n_alive = step_bodies(
            pos, mom, ori, amom, alive, sc, alive_idx, n_al,
            shape_id, 0.0005, 9.81,
            0.6, 0.5, 0.05, experiment.settle_height, steps)
        n_alive_box[0] = new_n_alive

        for nk in range(len(newly)):
            k = int(newly[nk])
            outcome = classify(shape_id,
                               ori[k, 0], ori[k, 1],
                               ori[k, 2], ori[k, 3])
            i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
            results[i2, j2] = outcome
            completed[0] += 1
            c = experiment.colors.get(int(outcome), "#999999")
            scene.mark_settled(k, c)

        if use_pyvista:
            try:
                scene.update_all(pos, ori, alive)
                settled_n = N - new_n_alive
                scene.set_title(
                    f"{experiment.shape} \u2014 {settled_n}/{N} settled "
                    f"\u2014 {new_n_alive} falling")
                scene.render()
            except Exception:
                pass
        else:
            scene.update_mpl_bodies(pos, ori, alive, results,
                                    grid_ij, experiment.colors)
            _update_scatter(scatter_data, N, grid_ij, pos, alive,
                            results, experiment.colors, ax_scatter,
                            new_n_alive)
            ax_scene.set_title(
                f"physical view \u2014 {new_n_alive} / {N} falling",
                fontsize=10)

        if len(newly) > 0:
            sweep_grid.update(img, results, experiment.colors)
            ax_map.set_title(
                f"{experiment.shape} outcome map \u2014 "
                f"{completed[0]} / {N} settled")
            category_histogram.update(bars, results, outcome_keys)

        if new_n_alive == 0:
            all_done[0] = True
            ax_map.set_title(
                f"{experiment.shape} outcome map \u2014 done")
            if use_pyvista:
                try:
                    scene.set_title(
                        f"{experiment.shape} \u2014 all {N} settled")
                except Exception:
                    pass

    ani = animation.FuncAnimation(fig, update, interval=30,
                                  cache_frame_data=False)
    plt.show()
    scene.close()


# ================================================================
# Static results mode
# ================================================================

def show_results(experiment, heights, angles, results,
                 tilt_axis="x", mode="", output_dir=None):
    """Static display of completed sweep results."""
    nh, na = len(heights), len(angles)
    outcome_keys = sorted(experiment.colors.keys())
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"width_ratios": [1.4, 0.6]})
    suffix = f" ({mode})" if mode else ""
    fig.suptitle(
        f"{experiment.shape} drop outcome map{suffix}",
        fontsize=13, fontweight="bold")

    img = sweep_grid.create(
        results, experiment.colors, extent,
        f"tilt about {tilt_axis}-axis (deg)", "height (m)",
        f"{experiment.shape} outcome map{suffix}", ax_map)
    sweep_grid.add_legend(ax_map, experiment.colors, experiment.labels)

    bars = category_histogram.create(
        outcome_keys, experiment.colors, experiment.labels, ax_hist)
    category_histogram.update(bars, results, outcome_keys)

    fig.tight_layout()

    if output_dir is not None:
        _save_figures(fig, output_dir)

    plt.show()


# ================================================================
# Replay mode
# ================================================================

def run_replay(experiment, heights, angles, results, tilt_axis="x"):
    """Animated replay of pre-computed results with 3D scene."""
    nh, na = len(heights), len(angles)
    N = nh * na
    shape_id = experiment.shape_id
    g = 9.81

    print("  Compiling physics (first run only)...", end=" ", flush=True)
    warmup(shape_id)
    print("done.", flush=True)

    settle_times = np.sqrt(2.0 * np.maximum(heights, 0.01) / g) * 6.0
    t_max = float(settle_times[-1])
    n_frames = max(nh, 200)
    time_axis = np.linspace(0, t_max, n_frames)
    dt_phys = 0.0005
    dt_frame = t_max / n_frames
    phys_steps_per_frame = max(1, int(dt_frame / dt_phys))
    if fast:
        print(f"  Fast video mode: {n_frames} frames, reduced DPI, "
              "subsampled 3D rendering.")

    ax_f, ay_f, az_f = _AXIS_MAP[tilt_axis]
    outcome_keys = sorted(experiment.colors.keys())
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    pos, mom, ori, amom, alive, sc, grid_ij = \
        _init_state(experiment, heights, angles, tilt_axis)
    offsets, spacing = _compute_offsets(nh, na, grid_ij)
    max_h = float(heights.max())

    use_pyvista = HAS_PYVISTA

    if use_pyvista:
        scene = BodyScene(
            experiment.mesh, N, offsets, max_h,
            body_color=experiment.body_color,
            title=f"{experiment.shape} drop \u2014 result viewer",
            use_pyvista=True)
        scene.show_nonblocking()

        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 0.6],
                              height_ratios=[1, 0.06],
                              hspace=0.30, wspace=0.22,
                              left=0.06, right=0.97, top=0.93, bottom=0.10)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_slider_pos = gs[1, 0]
    else:
        scene = BodyScene(
            experiment.mesh, N, offsets, max_h,
            body_color=experiment.body_color,
            use_pyvista=False)

        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 0.6],
                              height_ratios=[1, 0.06],
                              hspace=0.30, wspace=0.22,
                              left=0.04, right=0.97, top=0.93, bottom=0.10)

        ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
        ref_mesh, edge_lw, mesh_alpha = scene.init_mpl_scene(
            ax_scene, spacing)
        ax_scene.set_title(f"{experiment.shape}s \u2014 physical view")
        scene.create_mpl_bodies(ax_scene, pos, ori, edge_lw, mesh_alpha)

        ax_map = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[0, 2])
        ax_slider_pos = gs[1, :2]

    img = sweep_grid.create(
        results, experiment.colors, extent,
        f"tilt about {tilt_axis}-axis (deg)", "height (m)",
        f"{experiment.shape} outcome map", ax_map)
    sweep_grid.add_legend(ax_map, experiment.colors, experiment.labels)

    bars = category_histogram.create(
        outcome_keys, experiment.colors, experiment.labels, ax_hist)
    bar_counts = [int(np.sum(results == k)) for k in outcome_keys]
    for bar, c in zip(bars, bar_counts):
        bar.set_height(c)
    mx = max(1, max(bar_counts))
    ax_hist.set_ylim(0, mx * 1.15)

    from matplotlib.widgets import Slider
    ax_slider = fig.add_subplot(ax_slider_pos)
    slider = Slider(ax_slider, "time (s)", 0, n_frames - 1,
                    valinit=0, valstep=1, valfmt="%d")
    slider.valtext.set_text("t = 0.00 s")

    alive_idx = np.arange(N, dtype=np.int64)
    n_alive_box = [N]

    state = {"frame": 0, "playing": False, "updating": False,
             "phys_frame": 0,
             "play_wall_t0": 0.0, "play_sim_t0": 0.0}

    def _reset_physics():
        for si in range(N):
            i2, j2 = int(grid_ij[si, 0]), int(grid_ij[si, 1])
            pos[si] = [0.0, float(heights[i2]), 0.0]
            mom[si] = [0.0, 0.0, 0.0]
            w, x, y, z = quat_from_axis_angle(
                ax_f, ay_f, az_f, float(angles[j2]))
            ori[si] = [w, x, y, z]
            amom[si] = [0.0, 0.0, 0.0]
            alive[si] = True
            sc[si] = 0
        alive_idx[:N] = np.arange(N)
        n_alive_box[0] = N
        scene.reset()

    def _step_physics_to(target_frame):
        current = state["phys_frame"]
        if target_frame < current:
            _reset_physics()
            current = 0
        for f in range(current, target_frame):
            n_al = n_alive_box[0]
            if n_al == 0:
                break
            _, new_n = step_bodies(
                pos, mom, ori, amom, alive, sc,
                alive_idx, n_al,
                shape_id, dt_phys, g,
                0.6, 0.5, 0.05, experiment.settle_height,
                phys_steps_per_frame)
            n_alive_box[0] = new_n
        state["phys_frame"] = target_frame

    def _draw_frame(f):
        if state["updating"]:
            return
        state["updating"] = True
        f = max(0, min(f, n_frames - 1))
        state["frame"] = f
        t = time_axis[f]
        _step_physics_to(f)

        if use_pyvista:
            for si in range(N):
                if not alive[si]:
                    i2, j2 = int(grid_ij[si, 0]), int(grid_ij[si, 1])
                    val = results[i2, j2]
                    c = experiment.colors.get(int(val), "#999999")
                    scene.mark_settled(si, c)
            try:
                scene.update_all(pos, ori, alive)
                n_al = int(alive.sum())
                scene.set_title(
                    f"{experiment.shape}s \u2014 t = {t:.1f} s "
                    f"\u2014 {n_al} falling")
                scene.render()
            except Exception:
                pass
        else:
            scene.update_mpl_bodies(pos, ori, alive, results,
                                    grid_ij, experiment.colors)
            n_al = int(alive.sum())
            ax_scene.set_title(
                f"{experiment.shape}s \u2014 t = {t:.1f} s "
                f"\u2014 {n_al} falling")

        slider.set_val(f)
        slider.valtext.set_text(f"t = {t:.2f} s")
        fig.canvas.draw_idle()
        state["updating"] = False

    def on_slider(val):
        _draw_frame(int(val))

    def on_play():
        state["playing"] = True
        state["play_wall_t0"] = _time.monotonic()
        state["play_sim_t0"] = time_axis[state["frame"]]

    def on_pause(paused):
        if paused:
            state["playing"] = False
        else:
            on_play()

    def on_fwd():
        state["playing"] = False
        _draw_frame(state["frame"] + 1)

    def on_back():
        state["playing"] = False
        _draw_frame(state["frame"] - 1)

    slider.on_changed(on_slider)
    controls = PlaybackControls(
        fig, on_pause=on_pause, on_speed=lambda v: None,
        on_step_fwd=on_fwd, on_step_back=on_back)

    def _update(_anim_frame):
        if not state["playing"]:
            return
        if state["frame"] >= n_frames - 1:
            state["playing"] = False
            return
        elapsed = _time.monotonic() - state["play_wall_t0"]
        target_t = state["play_sim_t0"] + elapsed
        target_f = int(np.searchsorted(
            time_axis, target_t, side="right")) - 1
        target_f = max(state["frame"], min(target_f, n_frames - 1))
        if target_f != state["frame"]:
            _draw_frame(target_f)

    ani = animation.FuncAnimation(fig, _update, interval=33,
                                  cache_frame_data=False)
    _draw_frame(0)
    plt.show()
    scene.close()


# ================================================================
# Helpers
# ================================================================

def _set_flag(container, value):
    container[0] = value


def _setup_scatter_axes(ax, nh, na, spacing, max_h, shape, N):
    """Configure a 3D scatter axes for the fallback dashboard."""
    angle_spacing = {"coin": 0.45, "cube": 0.55}.get(shape, 0.6)
    h_spacing = {"coin": 0.35, "cube": 0.45}.get(shape, 0.4)
    off_x = np.arange(na, dtype=np.float64) * angle_spacing
    off_y = np.arange(nh, dtype=np.float64) * h_spacing
    off_x -= off_x.mean()
    off_y -= off_y.mean()

    ax.set_xlim(off_x[0] - 0.5, off_x[-1] + 0.5)
    ax.set_ylim(off_y[0] - 0.5, off_y[-1] + 0.5)
    ax.set_zlim(-0.1, max_h * 1.05)
    ax.set_xlabel("tilt angle", fontsize=8, labelpad=2)
    ax.set_ylabel("drop height h\u2080", fontsize=8, labelpad=2)
    ax.set_zlabel("height (m)", fontsize=9, labelpad=4)
    ax.set_title(f"all {N} dropping", fontsize=10)
    ax.view_init(elev=20, azim=-50)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


def _init_scatter(ax, N, grid_ij, nh, na, pos, spacing, colors):
    """Create and return the scatter plot data for fallback mode."""
    angle_spacing = 0.45
    h_spacing = 0.35
    off_x = np.arange(na, dtype=np.float64) * angle_spacing
    off_y = np.arange(nh, dtype=np.float64) * h_spacing
    off_x -= off_x.mean()
    off_y -= off_y.mean()

    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    ii = grid_ij[:, 0]
    jj = grid_ij[:, 1]
    xs[:] = off_x[jj]
    ys[:] = off_y[ii]
    zs[:] = pos[:, 1]

    marker_size = max(4, min(80, 3500 // max(1, N)))
    gray_rgba = np.array(mcolors.to_rgba("#999999"))
    init_colors = np.tile(gray_rgba, (N, 1))

    scatter = ax.scatter(xs, ys, zs, c=init_colors, s=marker_size,
                         alpha=0.85, depthshade=True, edgecolors="none")

    color_rgba = {}
    for v, hx in colors.items():
        color_rgba[v] = np.array(mcolors.to_rgba(hx))

    return {
        "scatter": scatter, "xs": xs, "ys": ys, "zs": zs,
        "off_x": off_x, "off_y": off_y,
        "gray_rgba": gray_rgba, "color_rgba": color_rgba,
    }


def _update_scatter(data, N, grid_ij, pos, alive, results,
                    colors, ax, n_alive):
    """Update the scatter plot positions and colours."""
    ii = grid_ij[:, 0]
    jj = grid_ij[:, 1]
    data["xs"][:] = data["off_x"][jj]
    data["ys"][:] = data["off_y"][ii]
    data["zs"][:] = np.clip(pos[:, 1], 0.0, None)

    rgba = np.tile(data["gray_rgba"], (N, 1))
    dead = ~alive
    if dead.any():
        for v, c in data["color_rgba"].items():
            mask = dead & (results[ii, jj] == v)
            rgba[mask] = c

    data["scatter"]._offsets3d = (
        data["xs"].copy(), data["ys"].copy(), data["zs"].copy())
    data["scatter"].set_facecolors(rgba)

    settled = N - n_alive
    ax.set_title(f"{settled} / {N} settled \u2014 {n_alive} falling",
                 fontsize=10)


def _save_figures(fig, output_dir):
    """Save figure PNGs to the output directory."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / "outcome_map.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out / 'outcome_map.png'}")


# ================================================================
# Video saving
# ================================================================

def save_video(experiment, heights, angles, results,
               tilt_axis="x", output_dir=None, fps=30, fast=False):
    """
    Save an animated replay of pre-computed results as an MP4.

    Uses matplotlib-only rendering (no PyVista) so that
    FuncAnimation.save() can capture every frame. Layout:
    3D mesh scene + outcome map + histogram, three panels.
    """
    from pathlib import Path
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nh, na = len(heights), len(angles)
    N = nh * na
    shape_id = experiment.shape_id
    g = 9.81

    print("  Compiling physics (first run only)...", end=" ", flush=True)
    warmup(shape_id)
    print("done.", flush=True)

    settle_times = np.sqrt(2.0 * np.maximum(heights, 0.01) / g) * 6.0
    t_max = float(settle_times[-1])
    if fast:
        n_frames = max(80, min(120, nh * 2))
    else:
        n_frames = max(nh, 200)
    time_axis = np.linspace(0, t_max, n_frames)
    dt_phys = 0.0005
    dt_frame = t_max / n_frames
    phys_steps_per_frame = max(1, int(dt_frame / dt_phys))

    ax_f, ay_f, az_f = _AXIS_MAP[tilt_axis]
    outcome_keys = sorted(experiment.colors.keys())
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]

    pos, mom, ori, amom, alive, sc, grid_ij = \
        _init_state(experiment, heights, angles, tilt_axis)
    offsets, spacing = _compute_offsets(nh, na, grid_ij)
    max_h = float(heights.max())

    from lab.visualization.body_scene import (
        BodyScene, _get_mesh_mpl, _transform_mesh,
    )
    ref_mesh = _get_mesh_mpl(experiment.mesh)

    fig = plt.figure(figsize=(14, 6) if fast else (18, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 0.5],
                          wspace=0.22, left=0.04, right=0.97,
                          top=0.90, bottom=0.08)

    ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
    nh_range = offsets[:, 1]
    na_range = offsets[:, 0]
    scene_w = (na_range.max() - na_range.min()) / 2 + 0.3
    scene_d = (nh_range.max() - nh_range.min()) / 2 + 0.3
    ax_scene.set_xlim(-scene_w, scene_w)
    ax_scene.set_ylim(-scene_d, scene_d)
    ax_scene.set_zlim(-0.05, max_h * 1.1)
    ax_scene.set_xlabel("x", fontsize=7)
    ax_scene.set_ylabel("z", fontsize=7)
    ax_scene.set_zlabel("height (m)", fontsize=8)
    ax_scene.set_title(f"{experiment.shape}s \u2014 physical view")
    ax_scene.view_init(elev=18, azim=-60)
    ax_scene.xaxis.set_tick_params(labelsize=6)
    ax_scene.yaxis.set_tick_params(labelsize=6)
    ax_scene.zaxis.set_tick_params(labelsize=7)

    fx = np.array([-scene_w, scene_w, scene_w, -scene_w])
    fz = np.array([-scene_d, -scene_d, scene_d, scene_d])
    fy = np.zeros(4)
    floor_poly = Poly3DCollection(
        [list(zip(fx, fz, fy))],
        alpha=0.15, facecolors="#888888",
        edgecolors="#aaaaaa", linewidths=0.5)
    ax_scene.add_collection3d(floor_poly)

    edge_lw = 0.25 if N > 50 else 0.35
    mesh_alpha = 0.70 if N > 100 else 0.80
    if fast and N > 4000:
        mesh_stride = int(np.ceil(N / 4000.0))
    else:
        mesh_stride = 1
    render_idx = np.arange(0, N, mesh_stride, dtype=np.int64)
    render_pos = {int(k): i for i, k in enumerate(render_idx)}
    mesh_collections = []
    for k in render_idx:
        ox, oz = offsets[k]
        faces = _transform_mesh(
            ref_mesh,
            ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
            ox, pos[k, 1], oz)
        pc = Poly3DCollection(
            faces, alpha=mesh_alpha,
            facecolors=experiment.body_color,
            edgecolors="#333333", linewidths=edge_lw)
        ax_scene.add_collection3d(pc)
        mesh_collections.append(pc)

    ax_map = fig.add_subplot(gs[0, 1])
    vid_results = np.full((nh, na), np.nan)
    img = sweep_grid.create(
        vid_results, experiment.colors, extent,
        f"tilt about {tilt_axis}-axis (deg)", "height (m)",
        f"{experiment.shape} outcome map", ax_map)
    sweep_grid.add_legend(ax_map, experiment.colors, experiment.labels)

    ax_hist = fig.add_subplot(gs[0, 2])
    bars = category_histogram.create(
        outcome_keys, experiment.colors, experiment.labels, ax_hist)

    title = (f"{experiment.shape} drop \u2014 {N} simulations "
             f"({nh}h \u00d7 {na}a)")
    if fast and mesh_stride > 1:
        title += f" \u2014 fast mode ({len(render_idx)} rendered)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    alive_idx = np.arange(N, dtype=np.int64)
    n_alive_box = [N]

    def _update(frame):
        n_al = n_alive_box[0]
        if n_al > 0:
            newly, new_n = step_bodies(
                pos, mom, ori, amom, alive, sc,
                alive_idx, n_al,
                shape_id, dt_phys, g,
                0.6, 0.5, 0.05, experiment.settle_height,
                phys_steps_per_frame)
            n_alive_box[0] = new_n

            for nk in range(len(newly)):
                k = int(newly[nk])
                outcome = classify(shape_id,
                                   ori[k, 0], ori[k, 1],
                                   ori[k, 2], ori[k, 3])
                i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
                vid_results[i2, j2] = outcome

            if len(newly) > 0 and ((not fast) or (frame % 2 == 0)):
                sweep_grid.update(img, vid_results, experiment.colors)
                category_histogram.update(bars, vid_results, outcome_keys)

        for k in render_idx:
            ox, oz = offsets[k]
            faces = _transform_mesh(
                ref_mesh,
                ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
                ox, pos[k, 1], oz)
            ci = render_pos[int(k)]
            mesh_collections[ci].set_verts(faces)
            if not alive[k]:
                i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
                val = vid_results[i2, j2]
                if not np.isnan(val):
                    c = experiment.colors.get(int(val), "#999999")
                    mesh_collections[ci].set_facecolors(c)

        t = time_axis[min(frame, len(time_axis) - 1)]
        n_al = n_alive_box[0]
        ax_scene.set_title(
            f"{experiment.shape}s \u2014 t = {t:.1f} s "
            f"\u2014 {n_al} falling", fontsize=10)

        settled = N - n_al
        ax_map.set_title(
            f"{experiment.shape} outcome map \u2014 "
            f"{settled} / {N} settled")

        progress_every = 10 if fast else 20
        if frame % progress_every == 0 or frame == n_frames - 1:
            print(f"\r  Encoding frame {frame + 1} / {n_frames}",
                  end="", flush=True)

        return mesh_collections + [img] + list(bars)

    ani = animation.FuncAnimation(fig, _update, frames=n_frames,
                                  interval=1000 // fps, blit=False)

    if output_dir is None:
        output_dir = "."
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    has_ffmpeg = animation.writers.is_available("ffmpeg")
    if not has_ffmpeg:
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = (
                imageio_ffmpeg.get_ffmpeg_exe())
            has_ffmpeg = True
        except ImportError:
            pass

    dpi = 70 if fast else 120
    if has_ffmpeg:
        video_path = out / "simulation.mp4"
        ani.save(str(video_path), writer="ffmpeg", fps=fps,
                 dpi=dpi, savefig_kwargs={"facecolor": fig.get_facecolor()})
    else:
        video_path = out / "simulation.gif"
        print("\n  ffmpeg not found, saving as GIF instead...")
        ani.save(str(video_path), writer="pillow", fps=fps, dpi=100)

    print(f"\n  Saved: {video_path}")
    plt.close(fig)
