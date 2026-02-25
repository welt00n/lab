"""
2D animations for pendulums, orbits, springs, particles, rigid bodies,
and generic trajectories.
"""

import numpy as np


def animate(dataset, interval=20, trail=True, save_path=None):
    """
    Dispatch to the appropriate animator based on the Hamiltonian's
    visualization hint.

    Falls back to a generic q-vs-time animation if no hint is given.
    """
    hint = dataset.hamiltonian.visualization_hint()
    anim_type = hint.get("type", "generic")

    if anim_type == "pendulum":
        return _animate_pendulum(dataset, hint, interval, save_path)
    elif anim_type == "double_pendulum":
        return _animate_double_pendulum(dataset, hint, interval, save_path)
    elif anim_type == "orbit":
        return _animate_orbit(dataset, hint, interval, trail, save_path)
    elif anim_type == "spring":
        return _animate_spring(dataset, hint, interval, save_path)
    elif anim_type == "coupled_spring":
        return _animate_coupled_spring(dataset, hint, interval, save_path)
    elif anim_type == "particle_3d":
        return _animate_particle_2d(dataset, hint, interval, save_path)
    elif anim_type == "rigid_drop":
        return _animate_rigid_drop(dataset, hint, interval, save_path)
    else:
        return _animate_generic(dataset, interval, save_path)


def _frames(n, cap=500):
    skip = max(1, n // cap)
    return range(0, n, skip)


def _convex_hull_2d(points):
    """Graham scan convex hull for 2D points. Returns vertices in order."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull) if hull else points


def _animate_pendulum(dataset, hint, interval, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    l = hint["lengths"][0]
    theta = dataset.q[:, 0]
    x = l * np.sin(theta)
    y = -l * np.cos(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-l * 1.3, l * 1.3)
    ax.set_ylim(-l * 1.3, l * 1.3)
    ax.set_aspect("equal")
    ax.set_title("Simple pendulum")
    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], "o-", color="steelblue", lw=2, markersize=8)
    trace, = ax.plot([], [], "-", color="steelblue", alpha=0.2, lw=1)
    trail_x, trail_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(frame):
        line.set_data([0, x[frame]], [0, y[frame]])
        trail_x.append(x[frame])
        trail_y.append(y[frame])
        trace.set_data(trail_x, trail_y)
        return line, trace

    frames = _frames(len(theta))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   init_func=init, interval=interval,
                                   blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_double_pendulum(dataset, hint, interval, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    l1, l2 = hint["lengths"]
    t1 = dataset.q[:, 0]
    t2 = dataset.q[:, 1]
    x1 = l1 * np.sin(t1)
    y1 = -l1 * np.cos(t1)
    x2 = x1 + l2 * np.sin(t2)
    y2 = y1 - l2 * np.cos(t2)

    fig, ax = plt.subplots(figsize=(6, 6))
    L = l1 + l2
    ax.set_xlim(-L * 1.2, L * 1.2)
    ax.set_ylim(-L * 1.2, L * 1.2)
    ax.set_aspect("equal")
    ax.set_title("Double pendulum")
    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], "o-", color="steelblue", lw=2, markersize=6)
    trace, = ax.plot([], [], "-", color="coral", alpha=0.3, lw=1)
    trail_x, trail_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(frame):
        line.set_data([0, x1[frame], x2[frame]],
                      [0, y1[frame], y2[frame]])
        trail_x.append(x2[frame])
        trail_y.append(y2[frame])
        trace.set_data(trail_x, trail_y)
        return line, trace

    frames = _frames(len(t1))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   init_func=init, interval=interval,
                                   blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_orbit(dataset, hint, interval, trail, save_path):
    """Animate an orbit in polar coordinates (r, theta) -> Cartesian."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    r = dataset.q[:, 0]
    theta = dataset.q[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = np.max(np.abs(np.concatenate([x, y]))) * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.plot(0, 0, "yo", markersize=10, label="central mass")
    ax.set_title("Kepler orbit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    point, = ax.plot([], [], "o", color="steelblue", markersize=6)
    trace, = ax.plot([], [], "-", color="steelblue", alpha=0.3, lw=1)
    trail_x, trail_y = [], []

    def update(frame):
        point.set_data([x[frame]], [y[frame]])
        if trail:
            trail_x.append(x[frame])
            trail_y.append(y[frame])
            trace.set_data(trail_x, trail_y)
        return point, trace

    frames = _frames(len(r))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_spring(dataset, hint, interval, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x = dataset.q[:, 0]
    t = dataset.t

    fig, (ax_spring, ax_trace) = plt.subplots(
        2, 1, figsize=(8, 5), height_ratios=[1, 1.5])

    lim = np.max(np.abs(x)) * 1.5
    ax_spring.set_xlim(-lim, lim)
    ax_spring.set_ylim(-0.5, 0.5)
    ax_spring.set_title("Harmonic oscillator")
    ax_spring.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_spring.set_yticks([])

    wall = plt.Rectangle((-lim - 0.05, -0.5), 0.05, 1.0,
                          fc="gray", ec="none", alpha=0.3)
    ax_spring.add_patch(wall)
    point, = ax_spring.plot([], [], "o", color="steelblue", markersize=14)

    ax_trace.set_xlim(t[0], t[-1])
    ax_trace.set_ylim(-lim, lim)
    ax_trace.set_xlabel("time")
    ax_trace.set_ylabel("x")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.plot(t, x, color="steelblue", alpha=0.2, lw=1)
    marker, = ax_trace.plot([], [], "o", color="steelblue", markersize=6)

    fig.tight_layout()

    def update(frame):
        point.set_data([x[frame]], [0])
        marker.set_data([t[frame]], [x[frame]])
        return point, marker

    frames = _frames(len(x))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_coupled_spring(dataset, hint, interval, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x1 = dataset.q[:, 0]
    x2 = dataset.q[:, 1]
    t = dataset.t

    fig, (ax_spring, ax_trace) = plt.subplots(
        2, 1, figsize=(9, 5), height_ratios=[1, 1.5])

    lim = max(np.max(np.abs(x1)), np.max(np.abs(x2))) * 2.0
    rest1, rest2 = -1.0, 1.0
    ax_spring.set_xlim(-lim - 1.5, lim + 1.5)
    ax_spring.set_ylim(-0.6, 0.6)
    ax_spring.set_title("Coupled oscillators")
    ax_spring.set_yticks([])

    p1, = ax_spring.plot([], [], "o", color="steelblue", markersize=14)
    p2, = ax_spring.plot([], [], "o", color="coral", markersize=14)
    spring_line, = ax_spring.plot([], [], "-", color="gray", lw=1.5, alpha=0.6)

    ax_trace.set_xlim(t[0], t[-1])
    ax_trace.set_ylim(-lim, lim)
    ax_trace.set_xlabel("time")
    ax_trace.set_ylabel("displacement")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.plot(t, x1, color="steelblue", alpha=0.2, lw=1, label="m1")
    ax_trace.plot(t, x2, color="coral", alpha=0.2, lw=1, label="m2")
    ax_trace.legend(loc="upper right", fontsize=8)
    m1_dot, = ax_trace.plot([], [], "o", color="steelblue", markersize=5)
    m2_dot, = ax_trace.plot([], [], "o", color="coral", markersize=5)

    fig.tight_layout()

    def update(frame):
        pos1 = rest1 + x1[frame]
        pos2 = rest2 + x2[frame]
        p1.set_data([pos1], [0])
        p2.set_data([pos2], [0])
        spring_line.set_data([pos1, pos2], [0, 0])
        m1_dot.set_data([t[frame]], [x1[frame]])
        m2_dot.set_data([t[frame]], [x2[frame]])
        return p1, p2, spring_line, m1_dot, m2_dot

    frames = _frames(len(x1))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_particle_2d(dataset, hint, interval, save_path):
    """Animate a charged particle as a top-down x-y trace with fading trail."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x = dataset.q[:, 0]
    y = dataset.q[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(dataset.hamiltonian.name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.plot(x[0], y[0], "x", color="gray", markersize=8, label="start")
    ax.legend(loc="upper right", fontsize=8)

    point, = ax.plot([], [], "o", color="#d62728", markersize=7)
    trace, = ax.plot([], [], "-", color="#d62728", alpha=0.25, lw=1)
    trail_x, trail_y = [], []

    def update(frame):
        point.set_data([x[frame]], [y[frame]])
        trail_x.append(x[frame])
        trail_y.append(y[frame])
        trace.set_data(trail_x, trail_y)
        return point, trace

    frames = _frames(len(x))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_rigid_drop(dataset, hint, interval, save_path):
    """
    Side-view animation of a rigid body falling, bouncing, and settling.

    Left panel: x-y side view with floor and rotating body cross-section.
    Right panel: height-vs-time trace.

    Shapes:
      - cube:  rotating square outline
      - coin:  rotating rectangle (edge-on, showing thickness)
      - rod:   rotating line segment with end-caps
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Polygon
    from lab.core import quaternion as quat

    pos_x = dataset.q[:, 0]
    pos_y = dataset.q[:, 1]
    orientations = dataset.q[:, 3:7]
    t = dataset.t

    shape = hint.get("shape", "cube")
    size = hint.get("size", 0.3)

    if shape == "cube":
        h = size / 2
        body_verts = np.array([
            [sx * h, sy * h, sz * h]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ])
    elif shape == "coin":
        r = size
        thick = 0.02
        n_rim = 12
        angles = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
        body_verts = np.concatenate([
            np.column_stack([r * np.cos(angles),
                             np.full(n_rim, -thick / 2),
                             r * np.sin(angles)]),
            np.column_stack([r * np.cos(angles),
                             np.full(n_rim, thick / 2),
                             r * np.sin(angles)]),
        ])
    elif shape == "rod":
        half = size / 2
        rad = 0.02
        body_verts = np.array([
            [-half, -rad, -rad], [-half, -rad, rad],
            [-half, rad, -rad], [-half, rad, rad],
            [half, -rad, -rad], [half, -rad, rad],
            [half, rad, -rad], [half, rad, rad],
        ])
    else:
        h = size / 2
        body_verts = np.array([
            [sx * h, sy * h, sz * h]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ])

    max_y = max(np.max(pos_y), 0.5)
    x_spread = max(np.max(np.abs(pos_x)) + size, size * 3)

    fig, (ax_scene, ax_trace) = plt.subplots(
        1, 2, figsize=(11, 6),
        gridspec_kw={"width_ratios": [1.3, 1]})

    ax_scene.set_xlim(-x_spread, x_spread)
    ax_scene.set_ylim(-0.15 * max_y, max_y * 1.15)
    ax_scene.set_aspect("equal")
    ax_scene.set_title(f"{shape} drop — side view", fontsize=12)
    ax_scene.set_xlabel("x  (m)")
    ax_scene.set_ylabel("height  (m)")

    floor_y = -0.15 * max_y
    ax_scene.axhline(0, color="#8B4513", lw=2.5, zorder=2)
    ax_scene.fill_between(
        [-x_spread * 1.5, x_spread * 1.5], floor_y, 0,
        color="#D2B48C", alpha=0.35, zorder=1)
    for xg in np.linspace(-x_spread, x_spread, 15):
        ax_scene.plot([xg, xg - 0.08 * max_y], [0, floor_y * 0.5],
                      color="#8B4513", lw=0.5, alpha=0.3, zorder=1)
    ax_scene.grid(True, alpha=0.15)

    body_patch = Polygon(
        body_verts[:, :2], closed=True,
        fc="steelblue", ec="midnightblue", lw=1.5, alpha=0.85, zorder=5)
    ax_scene.add_patch(body_patch)
    cm_dot, = ax_scene.plot([], [], "o", color="red", markersize=4, zorder=6)
    time_text = ax_scene.text(
        0.02, 0.96, "", transform=ax_scene.transAxes,
        fontsize=10, verticalalignment="top",
        fontfamily="monospace", color="0.3")

    ax_trace.set_xlim(t[0], t[-1])
    ax_trace.set_ylim(-0.05 * max_y, max_y * 1.05)
    ax_trace.set_xlabel("time  (s)")
    ax_trace.set_ylabel("height  (m)")
    ax_trace.set_title("CM height", fontsize=11)
    ax_trace.grid(True, alpha=0.2)
    ax_trace.axhline(0, color="#8B4513", lw=1, alpha=0.4)
    ax_trace.plot(t, pos_y, color="steelblue", alpha=0.2, lw=1)
    trace_dot, = ax_trace.plot([], [], "o", color="steelblue", markersize=5)
    trace_line, = ax_trace.plot([], [], "-", color="steelblue", lw=1.2, alpha=0.7)
    trace_tx, trace_ty = [], []

    fig.tight_layout(pad=1.5)

    def _rotated_verts_2d(frame):
        """Rotate all 3D body vertices and project onto x-y (side view).

        Returns the convex hull so overlapping interior points are
        discarded and the polygon renders as a proper silhouette.
        """
        q = orientations[frame]
        R = quat.to_rotation_matrix(q)
        cx, cy = pos_x[frame], pos_y[frame]

        world = (R @ body_verts.T).T
        pts = np.column_stack([world[:, 0] + cx, world[:, 1] + cy])
        return _convex_hull_2d(pts)

    def update(frame):
        verts = _rotated_verts_2d(frame)
        body_patch.set_xy(verts)
        cm_dot.set_data([pos_x[frame]], [pos_y[frame]])
        time_text.set_text(f"t = {t[frame]:.2f} s")

        trace_tx.append(t[frame])
        trace_ty.append(pos_y[frame])
        trace_line.set_data(trace_tx, trace_ty)
        trace_dot.set_data([t[frame]], [pos_y[frame]])

        return body_patch, cm_dot, time_text, trace_line, trace_dot

    frames = _frames(len(pos_x))
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def _animate_generic(dataset, interval, save_path):
    """Generic animation: sweep a marker along q_0(t)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.plot(dataset.t, dataset.q[:, 0], alpha=0.3)
    point, = ax.plot([], [], "o", color="red", markersize=8)
    ax.set_xlabel("time")
    ax.set_ylabel(dataset.hamiltonian.coords[0])
    ax.set_title(dataset.hamiltonian.name)
    ax.grid(True, alpha=0.3)

    def update(frame):
        point.set_data([dataset.t[frame]], [dataset.q[frame, 0]])
        return (point,)

    frames = _frames(dataset.nsteps)
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim
