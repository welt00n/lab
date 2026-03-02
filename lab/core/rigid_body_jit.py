"""
JIT-compiled rigid-body physics — single source of truth.

All constants, quaternion math, floor constraint, batch stepper, and
classification live here.  Every other module (dashboard, GPU wrapper,
OOP layer) imports from this file.

Compiled with Numba @njit for ~100-1000x speedup over pure Python.
Falls back to interpreted Python when Numba is not installed.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **_kw):
        if f is not None:
            return f
        return lambda func: func

# ================================================================
# Shape IDs
# ================================================================

COIN, CUBE, ROD = 0, 1, 2

SHAPE_NAME_TO_ID = {"coin": COIN, "cube": CUBE, "rod": ROD}

# ================================================================
# Physical constants (SI units)
# ================================================================

# Coin — US quarter (cupronickel, ~24 mm diameter)
COIN_RADIUS = 0.01213            # m
COIN_HALF_THICK = 0.000875       # m  (1.75 mm)
COIN_MASS = 0.00567              # kg (5.67 g)

# Die — standard 16 mm casino die (acrylic)
CUBE_HALF_SIDE = 0.008           # m  (16 mm side)
CUBE_MASS = 0.008                # kg (8 g)

# Rod (legacy, not exposed in experiments)
ROD_HALF_LEN = 0.5
ROD_RAD = 0.02
ROD_MASS = 1.0

# ================================================================
# JIT-compiled quaternion math (scalar arguments for Numba)
# ================================================================

@njit(cache=True)
def quat_normalize(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    s = 1.0 / n
    return w * s, x * s, y * s, z * s


@njit(cache=True)
def quat_multiply(w1, x1, y1, z1, w2, x2, y2, z2):
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


@njit(cache=True)
def quat_conjugate(w, x, y, z):
    return w, -x, -y, -z


@njit(cache=True)
def quat_rotate(qw, qx, qy, qz, vx, vy, vz):
    aw, ax, ay, az = quat_multiply(qw, qx, qy, qz, 0.0, vx, vy, vz)
    cw, cx, cy, cz = quat_conjugate(qw, qx, qy, qz)
    _, rx, ry, rz = quat_multiply(aw, ax, ay, az, cw, cx, cy, cz)
    return rx, ry, rz


@njit(cache=True)
def quat_exp_map(ox, oy, oz, dt):
    mag = math.sqrt(ox * ox + oy * oy + oz * oz)
    theta = mag * dt
    if theta < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    inv = 1.0 / mag
    half = theta * 0.5
    s = math.sin(half)
    return math.cos(half), s * ox * inv, s * oy * inv, s * oz * inv


@njit(cache=True)
def quat_from_axis_angle(ax, ay, az, angle):
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-15:
        return 1.0, 0.0, 0.0, 0.0
    inv = 1.0 / n
    half = angle * 0.5
    s = math.sin(half)
    return math.cos(half), s * ax * inv, s * ay * inv, s * az * inv


# ================================================================
# Vector helpers
# ================================================================

@njit(cache=True)
def cross(ax, ay, az, bx, by, bz):
    return ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx


@njit(cache=True)
def dot(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz


# ================================================================
# Lowest-point calculation (per shape)
# ================================================================

@njit(cache=True)
def _lowest_coin(qw, qx, qy, qz, py):
    r = COIN_RADIUS
    ht = COIN_HALF_THICK
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for sign in (-1.0, 1.0):
        wx, wy, wz = quat_rotate(qw, qx, qy, qz, 0.0, sign * ht, 0.0)
        wy2 = py + wy
        if wy2 < by_:
            by_ = wy2; bx_ = wx; bly = wy; blz = wz
    for k in range(8):
        a = k * 0.7853981633974483
        ca, sa = math.cos(a), math.sin(a)
        for sign in (-1.0, 1.0):
            wx, wy, wz = quat_rotate(qw, qx, qy, qz, r*ca, sign*ht, r*sa)
            wy2 = py + wy
            if wy2 < by_:
                by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def _lowest_cube(qw, qx, qy, qz, py):
    h = CUBE_HALF_SIDE
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                wx, wy, wz = quat_rotate(qw, qx, qy, qz, sx*h, sy*h, sz*h)
                wy2 = py + wy
                if wy2 < by_:
                    by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def _lowest_rod(qw, qx, qy, qz, py):
    half, rad = ROD_HALF_LEN, ROD_RAD
    by_ = 1e30
    bx_ = bly = blz = 0.0
    for ex in (-1.0, 1.0):
        for ey in (-1.0, 1.0):
            for ez in (-1.0, 1.0):
                wx, wy, wz = quat_rotate(qw, qx, qy, qz, ex*half, ey*rad, ez*rad)
                wy2 = py + wy
                if wy2 < by_:
                    by_ = wy2; bx_ = wx; bly = wy; blz = wz
    return by_, bx_, bly, blz


@njit(cache=True)
def lowest_point(shape, qw, qx, qy, qz, py):
    if shape == 0:
        return _lowest_coin(qw, qx, qy, qz, py)
    elif shape == 1:
        return _lowest_cube(qw, qx, qy, qz, py)
    else:
        return _lowest_rod(qw, qx, qy, qz, py)


# ================================================================
# Floor constraint (normal impulse + friction + rolling resistance)
# ================================================================

@njit(cache=True)
def floor_constraint(shape, mass, px, py, pz, mx, my, mz,
                     qw, qx, qy, qz, Lx, Ly, Lz, Ix, Iy, Iz,
                     rest, fric, rr):
    world_y, lx, ly, lz = lowest_point(shape, qw, qx, qy, qz, py)
    pen = world_y
    in_contact = pen < 0.0
    if shape == 0:
        near_floor = pen < 3.0 * COIN_RADIUS
    elif shape == 1:
        near_floor = pen < 3.0 * CUBE_HALF_SIDE
    else:
        near_floor = pen < 0.05

    if in_contact:
        py -= pen
        sIx = max(Ix, 1e-30)
        sIy = max(Iy, 1e-30)
        sIz = max(Iz, 1e-30)
        ox, oy, oz = Lx / sIx, Ly / sIy, Lz / sIz
        owx, owy, owz = quat_rotate(qw, qx, qy, qz, ox, oy, oz)
        vx, vy, vz = mx / mass, my / mass, mz / mass
        cx, cy, cz = cross(owx, owy, owz, lx, ly, lz)
        v_n = vy + cy

        if v_n < -0.01:
            cjw, cjx, cjy, cjz = quat_conjugate(qw, qx, qy, qz)
            rcn_x, rcn_y, rcn_z = cross(lx, ly, lz, 0.0, 1.0, 0.0)
            bx2, by2, bz2 = quat_rotate(cjw, cjx, cjy, cjz,
                                         rcn_x, rcn_y, rcn_z)
            ib_x, ib_y, ib_z = bx2 / sIx, by2 / sIy, bz2 / sIz
            iw_x, iw_y, iw_z = quat_rotate(qw, qx, qy, qz,
                                            ib_x, ib_y, ib_z)
            t_x, t_y, t_z = cross(iw_x, iw_y, iw_z, lx, ly, lz)
            inv_mass_eff = 1.0 / mass + t_y
            e_eff = rest if abs(v_n) > 0.1 else 0.0
            j_n = -(1.0 + e_eff) * v_n / inv_mass_eff

            my += j_n
            ti_x, ti_y, ti_z = cross(lx, ly, lz, 0.0, j_n, 0.0)
            tb_x, tb_y, tb_z = quat_rotate(cjw, cjx, cjy, cjz,
                                            ti_x, ti_y, ti_z)
            Lx += tb_x
            Ly += tb_y
            Lz += tb_z

            if fric > 0.0:
                ox2 = Lx / sIx
                oy2 = Ly / sIy
                oz2 = Lz / sIz
                owx2, owy2, owz2 = quat_rotate(qw, qx, qy, qz,
                                                ox2, oy2, oz2)
                vx2, vy2, vz2 = mx / mass, my / mass, mz / mass
                cx2, cy2, cz2 = cross(owx2, owy2, owz2, lx, ly, lz)
                vn2 = vy2 + cy2
                vt_x = vx2 + cx2
                vt_y = vy2 + cy2 - vn2
                vt_z = vz2 + cz2
                vt_mag = math.sqrt(vt_x*vt_x + vt_y*vt_y + vt_z*vt_z)
                if vt_mag > 1e-12:
                    iv = 1.0 / vt_mag
                    th_x, th_y, th_z = vt_x*iv, vt_y*iv, vt_z*iv
                    rct_x, rct_y, rct_z = cross(lx, ly, lz,
                                                th_x, th_y, th_z)
                    bt_x, bt_y, bt_z = quat_rotate(cjw, cjx, cjy, cjz,
                                                    rct_x, rct_y, rct_z)
                    ibt_x = bt_x / sIx
                    ibt_y = bt_y / sIy
                    ibt_z = bt_z / sIz
                    iwt_x, iwt_y, iwt_z = quat_rotate(qw, qx, qy, qz,
                                                       ibt_x, ibt_y, ibt_z)
                    tt_x, tt_y, tt_z = cross(iwt_x, iwt_y, iwt_z,
                                             lx, ly, lz)
                    rot_t = dot(tt_x, tt_y, tt_z, th_x, th_y, th_z)
                    ime_t = 1.0 / mass + rot_t
                    if ime_t < 1e-30:
                        ime_t = 1.0 / mass
                    jt = min(vt_mag / ime_t, fric * abs(j_n))
                    mx -= jt * th_x
                    my -= jt * th_y
                    mz -= jt * th_z
                    tf_x, tf_y, tf_z = cross(lx, ly, lz,
                                             -jt*th_x, -jt*th_y, -jt*th_z)
                    tfb_x, tfb_y, tfb_z = quat_rotate(cjw, cjx, cjy, cjz,
                                                       tf_x, tf_y, tf_z)
                    Lx += tfb_x
                    Ly += tfb_y
                    Lz += tfb_z

    if near_floor and rr > 0.0:
        if shape == 0:
            min_h = COIN_HALF_THICK
            char_size = COIN_RADIUS
        elif shape == 1:
            min_h = CUBE_HALF_SIDE
            char_size = CUBE_HALF_SIDE
        else:
            min_h = ROD_RAD
            char_size = ROD_RAD
        excess = max(0.0, py - min_h)
        stab = max(0.0, 1.0 - excess / (3.0 * char_size))
        eff = rr * stab
        if eff > 0.0:
            Lx *= (1.0 - eff)
            Ly *= (1.0 - eff)
            Lz *= (1.0 - eff)
        ke = ((mx*mx + my*my + mz*mz) / (2.0 * mass)
              + 0.5 * (Lx*Lx / max(Ix, 1e-30)
                       + Ly*Ly / max(Iy, 1e-30)
                       + Lz*Lz / max(Iz, 1e-30)))
        ke_scale = mass * 9.81 * char_size
        if excess < 2.0 * char_size and ke < 0.01 * ke_scale:
            mx = my = mz = 0.0
            Lx = Ly = Lz = 0.0
        elif ke < 0.5 * ke_scale:
            damp = 0.99
            mx *= damp
            my *= damp
            mz *= damp
            Lx *= damp
            Ly *= damp
            Lz *= damp

    return px, py, pz, mx, my, mz, qw, qx, qy, qz, Lx, Ly, Lz


# ================================================================
# Shape property helpers
# ================================================================

@njit(cache=True)
def get_mass(shape):
    if shape == 0:
        return COIN_MASS
    elif shape == 1:
        return CUBE_MASS
    else:
        return ROD_MASS


@njit(cache=True)
def get_inertia(shape):
    if shape == 0:
        m, r = COIN_MASS, COIN_RADIUS
        return m * 0.25 * r * r, m * 0.5 * r * r, m * 0.25 * r * r
    elif shape == 1:
        m, s = CUBE_MASS, 2.0 * CUBE_HALF_SIDE
        I = m * s * s / 6.0
        return I, I, I
    else:
        m, L = ROD_MASS, 2.0 * ROD_HALF_LEN
        Ip = m * L * L / 12.0
        return 1e-6 * m, Ip, Ip


# ================================================================
# Batch stepper — advances N bodies by n_steps
# ================================================================

@njit(cache=True)
def step_bodies(pos, mom, ori, amom, alive, sc, alive_idx, n_alive,
                shape, dt, g, rest, fric, rr, settle_h, n_steps):
    """
    Step alive bodies *n_steps* times.  Modifies arrays in-place.

    Uses a compact index array ``alive_idx[:n_alive]`` so the inner
    loop only touches living bodies.

    Returns (newly_settled_indices, updated_n_alive).
    """
    Ix, Iy, Iz = get_inertia(shape)
    mass = get_mass(shape)
    half = dt * 0.5
    inv_m = 1.0 / mass
    sIx = max(Ix, 1e-30)
    sIy = max(Iy, 1e-30)
    sIz = max(Iz, 1e-30)

    newly = np.empty(n_alive, dtype=np.int64)
    n_new = 0

    for _ in range(n_steps):
        write = 0
        for idx in range(n_alive):
            k = alive_idx[idx]

            mom[k, 1] -= mass * g * half

            pos[k, 0] += mom[k, 0] * inv_m * dt
            pos[k, 1] += mom[k, 1] * inv_m * dt
            pos[k, 2] += mom[k, 2] * inv_m * dt

            dw, dx, dy, dz = quat_exp_map(amom[k, 0] / sIx,
                                           amom[k, 1] / sIy,
                                           amom[k, 2] / sIz, dt)
            w2, x2, y2, z2 = quat_multiply(dw, dx, dy, dz,
                                            ori[k, 0], ori[k, 1],
                                            ori[k, 2], ori[k, 3])
            ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3] = \
                quat_normalize(w2, x2, y2, z2)

            (pos[k, 0], pos[k, 1], pos[k, 2],
             mom[k, 0], mom[k, 1], mom[k, 2],
             ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
             amom[k, 0], amom[k, 1], amom[k, 2]) = floor_constraint(
                shape, mass,
                pos[k, 0], pos[k, 1], pos[k, 2],
                mom[k, 0], mom[k, 1], mom[k, 2],
                ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
                amom[k, 0], amom[k, 1], amom[k, 2],
                Ix, Iy, Iz, rest, fric, rr)

            ke = ((mom[k, 0]**2 + mom[k, 1]**2 + mom[k, 2]**2)
                  * 0.5 * inv_m)
            ke += 0.5 * (amom[k, 0]**2 / sIx
                         + amom[k, 1]**2 / sIy
                         + amom[k, 2]**2 / sIz)

            ke_thr = mass * 9.81 * settle_h * 1e-4
            settled = False
            if ke < ke_thr and pos[k, 1] < settle_h:
                sc[k] += 1
                if sc[k] > 100:
                    settled = True
            elif pos[k, 1] < settle_h:
                sc[k] += 1
                if sc[k] > 5000:
                    settled = True
            else:
                sc[k] = 0

            if settled:
                alive[k] = False
                newly[n_new] = k
                n_new += 1
            else:
                alive_idx[write] = k
                write += 1

            mom[k, 1] -= mass * g * half

        n_alive = write

    return newly[:n_new], n_alive


# ================================================================
# Classification
# ================================================================

@njit(cache=True)
def classify(shape, qw, qx, qy, qz):
    if shape == 0:  # coin
        _, wy, _ = quat_rotate(qw, qx, qy, qz, 0.0, 1.0, 0.0)
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
        _, wy, _ = quat_rotate(qw, qx, qy, qz, 1.0, 0.0, 0.0)
        if wy < -0.1:
            return 1
        if wy > 0.1:
            return -1
        return 0


# ================================================================
# JIT warm-up helper
# ================================================================

def warmup(shape_id):
    """Trigger Numba compilation for the given shape. Cached on disk."""
    _w_pos = np.zeros((1, 3))
    _w_mom = np.zeros((1, 3))
    _w_ori = np.array([[1.0, 0.0, 0.0, 0.0]])
    _w_am = np.zeros((1, 3))
    _w_al = np.array([True])
    _w_sc = np.zeros(1, dtype=np.int64)
    _w_ai = np.zeros(1, dtype=np.int64)
    step_bodies(_w_pos, _w_mom, _w_ori, _w_am, _w_al, _w_sc, _w_ai, 1,
                shape_id, 0.0005, 9.81, 0.6, 0.5, 0.05, 0.5, 1)
    classify(shape_id, 1.0, 0.0, 0.0, 0.0)
