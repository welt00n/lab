"""
Live dashboard for drop experiments.

Three-panel FuncAnimation:
  Left:   3D scatter — every (height, angle) pair is an object in the
          scene, all dropping simultaneously.
  Centre: outcome map fills as objects settle.
  Right:  histogram updates in real time.

Physics are JIT-compiled with numba for speed.  Everything runs on the
main thread — no background workers, no queues, no deadlocks.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from lab.experiments.drop_experiment import (
    _results_to_rgb, _SHAPE_PALETTE,
)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **_kw):
        if f is not None:
            return f
        return lambda func: func

# Shape IDs (must match drop_gpu.py)
COIN, CUBE, ROD = 0, 1, 2
_SHAPE_TO_ID = {"coin": COIN, "cube": CUBE, "rod": ROD}

# ================================================================
# JIT-compiled quaternion math
# ================================================================

@njit(cache=True)
def _qn(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    s = 1.0 / n
    return w * s, x * s, y * s, z * s


@njit(cache=True)
def _qm(w1, x1, y1, z1, w2, x2, y2, z2):
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


@njit(cache=True)
def _qc(w, x, y, z):
    return w, -x, -y, -z


@njit(cache=True)
def _qr(qw, qx, qy, qz, vx, vy, vz):
    aw, ax, ay, az = _qm(qw, qx, qy, qz, 0.0, vx, vy, vz)
    cw, cx, cy, cz = _qc(qw, qx, qy, qz)
    _, rx, ry, rz = _qm(aw, ax, ay, az, cw, cx, cy, cz)
    return rx, ry, rz


@njit(cache=True)
def _qe(ox, oy, oz, dt):
    mag = math.sqrt(ox * ox + oy * oy + oz * oz)
    theta = mag * dt
    if theta < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    inv = 1.0 / mag
    half = theta * 0.5
    s = math.sin(half)
    return math.cos(half), s * ox * inv, s * oy * inv, s * oz * inv


@njit(cache=True)
def _qaa(ax, ay, az, angle):
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    inv = 1.0 / n
    half = angle * 0.5
    s = math.sin(half)
    return math.cos(half), s * ax * inv, s * ay * inv, s * az * inv


# ================================================================
# JIT-compiled lowest-point (per shape)
# ================================================================

@njit(cache=True)
def _cross(ax, ay, az, bx, by, bz):
    return ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx


@njit(cache=True)
def _dot(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz


@njit(cache=True)
def _lowest_coin(qw, qx, qy, qz, py):
    r, ht = 0.15, 0.01
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for sign in (-1.0, 1.0):
        wx, wy, wz = _qr(qw, qx, qy, qz, 0.0, sign * ht, 0.0)
        wy2 = py + wy
        if wy2 < by_:
            by_ = wy2; bx_ = wx; bly = wy; blz = wz
    for k in range(8):
        a = k * 0.7853981633974483
        ca, sa = math.cos(a), math.sin(a)
        for sign in (-1.0, 1.0):
            wx, wy, wz = _qr(qw, qx, qy, qz, r*ca, sign*ht, r*sa)
            wy2 = py + wy
            if wy2 < by_:
                by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def _lowest_cube(qw, qx, qy, qz, py):
    h = 0.15
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                wx, wy, wz = _qr(qw, qx, qy, qz, sx*h, sy*h, sz*h)
                wy2 = py + wy
                if wy2 < by_:
                    by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def _lowest_rod(qw, qx, qy, qz, py):
    half, rad = 0.5, 0.02
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for ex in (-1.0, 1.0):
        for ey in (-1.0, 1.0):
            for ez in (-1.0, 1.0):
                wx, wy, wz = _qr(qw, qx, qy, qz, ex*half, ey*rad, ez*rad)
                wy2 = py + wy
                if wy2 < by_:
                    by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def _lowest(shape, qw, qx, qy, qz, py):
    if shape == 0:
        return _lowest_coin(qw, qx, qy, qz, py)
    elif shape == 1:
        return _lowest_cube(qw, qx, qy, qz, py)
    else:
        return _lowest_rod(qw, qx, qy, qz, py)


# ================================================================
# JIT-compiled floor constraint
# ================================================================

@njit(cache=True)
def _floor(shape, mass, px, py, pz, mx, my, mz,
           qw, qx, qy, qz, Lx, Ly, Lz, Ix, Iy, Iz,
           rest, fric, rr):
    world_y, lx, ly, lz = _lowest(shape, qw, qx, qy, qz, py)
    pen = world_y
    in_contact = pen < 0.0
    near_floor = pen < 0.05

    if in_contact:
        py -= pen
        sIx = max(Ix, 1e-30)
        sIy = max(Iy, 1e-30)
        sIz = max(Iz, 1e-30)
        ox, oy, oz = Lx / sIx, Ly / sIy, Lz / sIz
        owx, owy, owz = _qr(qw, qx, qy, qz, ox, oy, oz)
        vx, vy, vz = mx / mass, my / mass, mz / mass
        cx, cy, cz = _cross(owx, owy, owz, lx, ly, lz)
        v_n = vy + cy

        if v_n < -0.01:
            cjw, cjx, cjy, cjz = _qc(qw, qx, qy, qz)
            rcn_x, rcn_y, rcn_z = _cross(lx, ly, lz, 0.0, 1.0, 0.0)
            bx2, by2, bz2 = _qr(cjw, cjx, cjy, cjz, rcn_x, rcn_y, rcn_z)
            ib_x, ib_y, ib_z = bx2 / sIx, by2 / sIy, bz2 / sIz
            iw_x, iw_y, iw_z = _qr(qw, qx, qy, qz, ib_x, ib_y, ib_z)
            t_x, t_y, t_z = _cross(iw_x, iw_y, iw_z, lx, ly, lz)
            inv_mass_eff = 1.0 / mass + t_y
            e_eff = rest if abs(v_n) > 0.1 else 0.0
            j_n = -(1.0 + e_eff) * v_n / inv_mass_eff

            my += j_n
            ti_x, ti_y, ti_z = _cross(lx, ly, lz, 0.0, j_n, 0.0)
            tb_x, tb_y, tb_z = _qr(cjw, cjx, cjy, cjz, ti_x, ti_y, ti_z)
            Lx += tb_x
            Ly += tb_y
            Lz += tb_z

            if fric > 0.0:
                ox2 = Lx / sIx
                oy2 = Ly / sIy
                oz2 = Lz / sIz
                owx2, owy2, owz2 = _qr(qw, qx, qy, qz, ox2, oy2, oz2)
                vx2, vy2, vz2 = mx / mass, my / mass, mz / mass
                cx2, cy2, cz2 = _cross(owx2, owy2, owz2, lx, ly, lz)
                vn2 = vy2 + cy2
                vt_x = vx2 + cx2
                vt_y = vy2 + cy2 - vn2
                vt_z = vz2 + cz2
                vt_mag = math.sqrt(vt_x*vt_x + vt_y*vt_y + vt_z*vt_z)
                if vt_mag > 1e-12:
                    iv = 1.0 / vt_mag
                    th_x, th_y, th_z = vt_x*iv, vt_y*iv, vt_z*iv
                    rct_x, rct_y, rct_z = _cross(lx, ly, lz,
                                                  th_x, th_y, th_z)
                    bt_x, bt_y, bt_z = _qr(cjw, cjx, cjy, cjz,
                                            rct_x, rct_y, rct_z)
                    ibt_x = bt_x / sIx
                    ibt_y = bt_y / sIy
                    ibt_z = bt_z / sIz
                    iwt_x, iwt_y, iwt_z = _qr(qw, qx, qy, qz,
                                                ibt_x, ibt_y, ibt_z)
                    tt_x, tt_y, tt_z = _cross(iwt_x, iwt_y, iwt_z,
                                               lx, ly, lz)
                    rot_t = _dot(tt_x, tt_y, tt_z, th_x, th_y, th_z)
                    ime_t = 1.0 / mass + rot_t
                    if ime_t < 1e-30:
                        ime_t = 1.0 / mass
                    jt = min(vt_mag / ime_t, fric * abs(j_n))
                    mx -= jt * th_x
                    my -= jt * th_y
                    mz -= jt * th_z
                    tf_x, tf_y, tf_z = _cross(lx, ly, lz,
                                               -jt*th_x, -jt*th_y, -jt*th_z)
                    tfb_x, tfb_y, tfb_z = _qr(cjw, cjx, cjy, cjz,
                                                tf_x, tf_y, tf_z)
                    Lx += tfb_x
                    Ly += tfb_y
                    Lz += tfb_z

    if near_floor and rr > 0.0:
        if shape == 0:
            min_h = 0.01
        elif shape == 1:
            min_h = 0.15
        else:
            min_h = 0.02
        excess = max(0.0, py - min_h)
        stab = max(0.0, 1.0 - excess / 0.05)
        eff = rr * stab
        if eff > 0.0:
            Lx *= (1.0 - eff)
            Ly *= (1.0 - eff)
            Lz *= (1.0 - eff)
        ke = ((mx*mx + my*my + mz*mz) / (2.0 * mass)
              + 0.5 * (Lx*Lx / max(Ix, 1e-30)
                       + Ly*Ly / max(Iy, 1e-30)
                       + Lz*Lz / max(Iz, 1e-30)))
        if excess < 0.05 and ke < 0.1:
            mx = my = mz = 0.0
            Lx = Ly = Lz = 0.0
        elif ke < 5.0:
            damp = 0.99
            mx *= damp
            my *= damp
            mz *= damp
            Lx *= damp
            Ly *= damp
            Lz *= damp

    return px, py, pz, mx, my, mz, qw, qx, qy, qz, Lx, Ly, Lz


# ================================================================
# JIT-compiled main stepper
# ================================================================

@njit(cache=True)
def _get_inertia(shape):
    if shape == 0:
        r = 0.15
        return 0.25 * r * r, 0.5 * r * r, 0.25 * r * r
    elif shape == 1:
        s = 0.3
        I = s * s / 6.0
        return I, I, I
    else:
        L = 1.0
        Ip = L * L / 12.0
        return 1e-6, Ip, Ip


@njit(cache=True)
def _step_all(pos, mom, ori, amom, alive, sc,
              shape, dt, g, rest, fric, rr, settle_h, n_steps):
    """
    Step all alive bodies *n_steps* times.  Modifies arrays in-place.
    Returns an int64 array of body indices that just settled.
    """
    N = pos.shape[0]
    Ix, Iy, Iz = _get_inertia(shape)
    mass = 1.0
    half = dt * 0.5
    newly = np.empty(N, dtype=np.int64)
    n_new = 0

    for _ in range(n_steps):
        for k in range(N):
            if not alive[k]:
                continue

            # Half kick (gravity)
            mom[k, 1] -= mass * g * half

            # Drift
            inv_m = 1.0 / mass
            pos[k, 0] += mom[k, 0] * inv_m * dt
            pos[k, 1] += mom[k, 1] * inv_m * dt
            pos[k, 2] += mom[k, 2] * inv_m * dt

            # Quaternion drift
            sIx = max(Ix, 1e-30)
            sIy = max(Iy, 1e-30)
            sIz = max(Iz, 1e-30)
            dw, dx, dy, dz = _qe(amom[k, 0] / sIx,
                                  amom[k, 1] / sIy,
                                  amom[k, 2] / sIz, dt)
            w2, x2, y2, z2 = _qm(dw, dx, dy, dz,
                                   ori[k, 0], ori[k, 1],
                                   ori[k, 2], ori[k, 3])
            ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3] = _qn(w2, x2, y2, z2)

            # Floor constraint
            (pos[k, 0], pos[k, 1], pos[k, 2],
             mom[k, 0], mom[k, 1], mom[k, 2],
             ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
             amom[k, 0], amom[k, 1], amom[k, 2]) = _floor(
                shape, mass,
                pos[k, 0], pos[k, 1], pos[k, 2],
                mom[k, 0], mom[k, 1], mom[k, 2],
                ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
                amom[k, 0], amom[k, 1], amom[k, 2],
                Ix, Iy, Iz, rest, fric, rr)

            # Settle detection (before second half-kick so the floor
            # constraint's zero-momentum state is visible)
            ke = ((mom[k, 0]**2 + mom[k, 1]**2 + mom[k, 2]**2)
                  * 0.5 * inv_m)
            ke += 0.5 * (amom[k, 0]**2 / sIx
                         + amom[k, 1]**2 / sIy
                         + amom[k, 2]**2 / sIz)

            if ke < 1e-4 and pos[k, 1] < settle_h:
                sc[k] += 1
                if sc[k] > 100:
                    alive[k] = False
                    newly[n_new] = k
                    n_new += 1
            elif pos[k, 1] < settle_h:
                sc[k] += 1
                if sc[k] > 5000:
                    alive[k] = False
                    newly[n_new] = k
                    n_new += 1
            else:
                sc[k] = 0

            # Second half kick
            mom[k, 1] -= mass * g * half

    return newly[:n_new]


# ================================================================
# JIT-compiled classification
# ================================================================

@njit(cache=True)
def _classify_jit(shape, qw, qx, qy, qz):
    if shape == 0:  # coin
        _, wy, _ = _qr(qw, qx, qy, qz, 0.0, 1.0, 0.0)
        if wy > 0.1:
            return 1
        if wy < -0.1:
            return -1
        return 0
    elif shape == 1:  # cube
        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)
        mi = 0
        mv = r10
        if -r10 < mv:
            mv = -r10; mi = 1
        if r11 < mv:
            mv = r11; mi = 2
        if -r11 < mv:
            mv = -r11; mi = 3
        if r12 < mv:
            mv = r12; mi = 4
        if -r12 < mv:
            mi = 5
        return mi
    else:  # rod
        _, wy, _ = _qr(qw, qx, qy, qz, 1.0, 0.0, 0.0)
        if wy < -0.1:
            return 1
        if wy > 0.1:
            return -1
        return 0


# ================================================================
# Dashboard
# ================================================================

def run_live_dashboard(shape, heights, angles, tilt_axis="x",
                       workers=None, gpu=False):
    """
    Three-panel live dashboard.

    Every grid point is a dot in the 3D scatter.  All fall together in
    real time.  The outcome map and histogram fill as objects settle.
    """
    nh, na = len(heights), len(angles)
    N = nh * na
    shape_id = _SHAPE_TO_ID[shape]
    colors_map, label_map = _SHAPE_PALETTE[shape]

    if N > 500:
        print(f"  NOTE: {N} objects is heavy for live mode. "
              f"Consider --nh 10 --na 15 for smoother animation.")

    # Axis direction
    ax_f, ay_f, az_f = {"x": (1., 0., 0.),
                        "y": (0., 1., 0.),
                        "z": (0., 0., 1.)}[tilt_axis]

    # --- Initialize state arrays ---
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
            w, x, y, z = _qaa(ax_f, ay_f, az_f, float(a))
            ori[idx] = [w, x, y, z]
            grid_ij[idx] = [i, j]
            idx += 1

    settle_h = {"coin": 0.20, "cube": 0.35, "rod": 0.55}.get(shape, 0.5)
    results = np.full((nh, na), np.nan)

    # --- JIT warm-up (compile before animation starts) ---
    print("  Compiling physics (first run only)...", end=" ", flush=True)
    _w_pos = np.zeros((1, 3))
    _w_mom = np.zeros((1, 3))
    _w_ori = np.array([[1.0, 0.0, 0.0, 0.0]])
    _w_am = np.zeros((1, 3))
    _w_al = np.array([True])
    _w_sc = np.zeros(1, dtype=np.int64)
    _step_all(_w_pos, _w_mom, _w_ori, _w_am, _w_al, _w_sc,
              shape_id, 0.001, 9.81, 0.6, 0.5, 0.05, 0.5, 1)
    _classify_jit(shape_id, 1.0, 0.0, 0.0, 0.0)
    print("done.", flush=True)

    # --- Spatial offsets ---
    # x-axis: spread by tilt angle, y-axis: spread by drop height, z-axis: real height
    angle_spacing = {"coin": 0.45, "cube": 0.55, "rod": 1.2}.get(shape, 0.6)
    h_spacing = {"coin": 0.35, "cube": 0.45, "rod": 0.8}.get(shape, 0.4)
    off_x = np.arange(na, dtype=np.float64) * angle_spacing
    off_y = np.arange(nh, dtype=np.float64) * h_spacing
    off_x -= off_x.mean()
    off_y -= off_y.mean()

    # --- Figure ---
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.2, 0.6], wspace=0.22)

    ax3d = fig.add_subplot(gs[0], projection="3d")
    max_h = float(heights.max())
    ax3d.set_xlim(off_x[0] - 0.5, off_x[-1] + 0.5)
    ax3d.set_ylim(off_y[0] - 0.5, off_y[-1] + 0.5)
    ax3d.set_zlim(-0.1, max_h * 1.05)
    ax3d.set_xlabel("tilt angle", fontsize=8, labelpad=2)
    ax3d.set_ylabel("drop height h\u2080", fontsize=8, labelpad=2)
    ax3d.set_zlabel("height (m)", fontsize=9, labelpad=4)
    ax3d.set_title(f"{N} {shape}s dropping")
    ax3d.view_init(elev=20, azim=-50)
    ax3d.xaxis.set_tick_params(labelsize=6)
    ax3d.yaxis.set_tick_params(labelsize=6)
    ax3d.zaxis.set_tick_params(labelsize=7)
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax3d.yaxis.set_major_locator(plt.MaxNLocator(5))

    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    ii_init = grid_ij[:, 0]
    jj_init = grid_ij[:, 1]
    xs[:] = off_x[jj_init]
    ys[:] = off_y[ii_init]
    zs[:] = pos[:, 1]
    marker_size = max(4, min(80, 3500 // max(1, N)))

    gray_rgba = np.array(mcolors.to_rgba("#999999"))
    color_rgba = {}
    for v, hx in colors_map.items():
        color_rgba[v] = np.array(mcolors.to_rgba(hx))

    init_colors = np.tile(gray_rgba, (N, 1))
    scatter = ax3d.scatter(xs, ys, zs, c=init_colors, s=marker_size,
                           alpha=0.85, depthshade=True, edgecolors="none")

    # Outcome map
    ax_map = fig.add_subplot(gs[1])
    rgb = _results_to_rgb(results, colors_map)
    extent = [np.degrees(angles[0]), np.degrees(angles[-1]),
              heights[0], heights[-1]]
    img = ax_map.imshow(rgb, origin="lower", aspect="auto", extent=extent,
                        interpolation="nearest")
    ax_map.set_xlabel(f"tilt about {tilt_axis}-axis (deg)")
    ax_map.set_ylabel("height (m)")
    ax_map.set_title(f"{shape} — 0% settled")

    # Histogram
    ax_hist = fig.add_subplot(gs[2])
    outcome_keys = sorted(colors_map.keys())
    bar_colors = [colors_map[k] for k in outcome_keys]
    bar_labels = [label_map[k] for k in outcome_keys]
    bars = ax_hist.bar(range(len(outcome_keys)), [0] * len(outcome_keys),
                       color=bar_colors)
    ax_hist.set_xticks(range(len(outcome_keys)))
    ax_hist.set_xticklabels(bar_labels, fontsize=8, rotation=30)
    ax_hist.set_ylabel("count")
    ax_hist.set_title("distribution")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08,
                        top=0.92, wspace=0.22)

    # --- Animation state ---
    completed = [0]
    steps_per_frame = max(10, min(150, 5000 // max(1, N)))
    all_done = [False]

    def update(_frame):
        if all_done[0]:
            return (scatter, img) + tuple(bars)

        newly = _step_all(pos, mom, ori, amom, alive, sc,
                          shape_id, 0.001, 9.81,
                          0.6, 0.5, 0.05, settle_h, steps_per_frame)

        # Classify newly settled bodies
        for nk in range(len(newly)):
            k = int(newly[nk])
            outcome = _classify_jit(shape_id,
                                    ori[k, 0], ori[k, 1],
                                    ori[k, 2], ori[k, 3])
            i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
            results[i2, j2] = outcome
            completed[0] += 1

        # Update map & histogram
        if len(newly) > 0:
            rgb_new = _results_to_rgb(results, colors_map)
            img.set_data(rgb_new)
            pct = 100 * completed[0] / N
            ax_map.set_title(
                f"{shape} — {pct:.0f}% settled ({completed[0]}/{N})")
            for bi, key in enumerate(outcome_keys):
                bars[bi].set_height(int(np.nansum(results == key)))
            mx = max(1, max(int(np.nansum(results == k))
                            for k in outcome_keys))
            ax_hist.set_ylim(0, mx * 1.15)

        # Update scatter positions (vectorized)
        # x = angle spread + lateral drift, y = drop-height spread, z = actual height
        ii = grid_ij[:, 0]
        jj = grid_ij[:, 1]
        xs[:] = pos[:, 0] + off_x[jj]
        ys[:] = pos[:, 2] + off_y[ii]
        zs[:] = pos[:, 1]

        # Update colours (vectorized)
        rgba = np.tile(gray_rgba, (N, 1))
        dead = ~alive
        if dead.any():
            for v, c in color_rgba.items():
                mask = dead & (results[ii, jj] == v)
                rgba[mask] = c

        scatter._offsets3d = (xs.copy(), ys.copy(), zs.copy())
        scatter.set_facecolors(rgba)

        n_alive = int(alive.sum())
        ax3d.set_title(f"{N} {shape}s — {n_alive} still falling")

        if n_alive == 0:
            all_done[0] = True
            pct = 100 * completed[0] / N
            ax_map.set_title(
                f"{shape} — done ({completed[0]}/{N})")

        return (scatter, img) + tuple(bars)

    ani = animation.FuncAnimation(fig, update, interval=30,
                                  cache_frame_data=False)
    plt.show()
