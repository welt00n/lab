"""
GPU-accelerated drop experiment using Numba CUDA.

Reimplements the rigid-body drop simulation as pure numeric CUDA device
functions so that the entire height x angle parameter grid can run in
parallel — one CUDA thread per simulation.

Requires:
    numba, nvidia-cuda-nvcc-cu12, nvidia-cuda-runtime-cu12

Environment variables (set automatically by ``_setup_cuda_env``):
    CUDA_HOME  — points to the pip-installed cuda_nvcc package
    LD_LIBRARY_PATH — includes libnvvm and libcudart directories
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------
# Auto-configure CUDA library paths from pip-installed packages
# ------------------------------------------------------------------

def _setup_cuda_env():
    """Point numba at the pip-installed CUDA toolkit libraries."""
    try:
        import nvidia.cuda_nvcc as _nvcc
        import nvidia.cuda_runtime as _rt
    except ImportError:
        return

    nvcc_root = str(Path(_nvcc.__file__).resolve().parent)
    rt_root = str(Path(_rt.__file__).resolve().parent)

    nvvm_lib = os.path.join(nvcc_root, "nvvm", "lib64")
    cudart_lib = os.path.join(rt_root, "lib")

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = ld.split(":") if ld else []
    for d in (nvvm_lib, cudart_lib):
        if d not in parts:
            parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = nvcc_root

_setup_cuda_env()


# ------------------------------------------------------------------
# Import numba.cuda (after env setup)
# ------------------------------------------------------------------

def _cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False

HAS_CUDA = _cuda_available()

if HAS_CUDA:
    from numba import cuda, float64, int32
else:
    cuda = None


# ------------------------------------------------------------------
# Shape IDs
# ------------------------------------------------------------------

SHAPE_COIN = 0
SHAPE_CUBE = 1
SHAPE_ROD = 2

_SHAPE_NAME_TO_ID = {"coin": SHAPE_COIN, "cube": SHAPE_CUBE, "rod": SHAPE_ROD}


# ====================================================================
# CUDA device functions
# ====================================================================

if HAS_CUDA:

    # ---------------------------------------------------------------
    # Quaternion math
    # ---------------------------------------------------------------

    @cuda.jit(device=True)
    def quat_normalize(w, x, y, z):
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-15:
            return 1.0, 0.0, 0.0, 0.0
        inv = 1.0 / n
        return w * inv, x * inv, y * inv, z * inv

    @cuda.jit(device=True)
    def quat_multiply(w1, x1, y1, z1, w2, x2, y2, z2):
        return (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    @cuda.jit(device=True)
    def quat_conjugate(w, x, y, z):
        return w, -x, -y, -z

    @cuda.jit(device=True)
    def quat_rotate_vector(qw, qx, qy, qz, vx, vy, vz):
        """Rotate vector (vx,vy,vz) by quaternion q:  q * (0,v) * q*."""
        aw, ax, ay, az = quat_multiply(qw, qx, qy, qz, 0.0, vx, vy, vz)
        cw, cx, cy, cz = quat_conjugate(qw, qx, qy, qz)
        _, rx, ry, rz = quat_multiply(aw, ax, ay, az, cw, cx, cy, cz)
        return rx, ry, rz

    @cuda.jit(device=True)
    def quat_from_axis_angle(ax, ay, az, angle):
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-15:
            return 1.0, 0.0, 0.0, 0.0
        inv = 1.0 / n
        ax2 = ax * inv
        ay2 = ay * inv
        az2 = az * inv
        half = angle * 0.5
        s = math.sin(half)
        return math.cos(half), s * ax2, s * ay2, s * az2

    @cuda.jit(device=True)
    def quat_exp_map(ox, oy, oz, dt):
        """Quaternion from angular velocity omega over timestep dt."""
        mag = math.sqrt(ox * ox + oy * oy + oz * oz)
        theta = mag * dt
        if theta < 1e-15:
            return 1.0, 0.0, 0.0, 0.0
        inv = 1.0 / mag
        half = theta * 0.5
        s = math.sin(half)
        return math.cos(half), s * ox * inv, s * oy * inv, s * oz * inv

    # ---------------------------------------------------------------
    # Rotation matrix (for cube classification)
    # ---------------------------------------------------------------

    @cuda.jit(device=True)
    def quat_to_rotmat_row(qw, qx, qy, qz, row):
        """Return one row of the 3x3 rotation matrix (row 0, 1, or 2)."""
        if row == 0:
            return (1.0 - 2.0*(qy*qy + qz*qz),
                    2.0*(qx*qy - qz*qw),
                    2.0*(qx*qz + qy*qw))
        elif row == 1:
            return (2.0*(qx*qy + qz*qw),
                    1.0 - 2.0*(qx*qx + qz*qz),
                    2.0*(qy*qz - qx*qw))
        else:
            return (2.0*(qx*qz - qy*qw),
                    2.0*(qy*qz + qx*qw),
                    1.0 - 2.0*(qx*qx + qy*qy))

    # ---------------------------------------------------------------
    # Body shape parameters (hard-coded, no Python objects)
    # ---------------------------------------------------------------
    # Coin: mass=1, radius=0.15, thickness=0.02
    # Cube: mass=1, side=0.3
    # Rod:  mass=1, length=1.0, radius=0.02

    @cuda.jit(device=True)
    def get_inertia(shape_id):
        """Return (Ix, Iy, Iz) for the shape."""
        if shape_id == SHAPE_COIN:
            r = 0.15
            Ix = 0.25 * r * r          # m*r^2/4
            Iy = 0.5 * r * r           # m*r^2/2
            Iz = 0.25 * r * r
            return Ix, Iy, Iz
        elif shape_id == SHAPE_CUBE:
            s = 0.3
            I = s * s / 6.0
            return I, I, I
        else:
            L = 1.0
            Ip = L * L / 12.0
            return 1e-6, Ip, Ip

    @cuda.jit(device=True)
    def get_min_rest_height(shape_id):
        """Minimum CM height when resting on the most stable face."""
        if shape_id == SHAPE_COIN:
            return 0.01      # half-thickness
        elif shape_id == SHAPE_CUBE:
            return 0.15      # side/2
        else:
            return 0.0       # rod on floor has CM at 0

    # ---------------------------------------------------------------
    # Lowest point calculation
    # ---------------------------------------------------------------

    @cuda.jit(device=True)
    def lowest_point_coin(qw, qx, qy, qz, pos_y):
        """Return (min_world_y, lever_x, lever_y, lever_z)."""
        r = 0.15
        ht = 0.01  # half thickness
        best_y = 1e30
        best_lx = 0.0
        best_ly = 0.0
        best_lz = 0.0

        # Face centers
        for sign in (-1.0, 1.0):
            bx, by, bz = 0.0, sign * ht, 0.0
            wx, wy, wz = quat_rotate_vector(qw, qx, qy, qz, bx, by, bz)
            world_y = pos_y + wy
            if world_y < best_y:
                best_y = world_y
                best_lx, best_ly, best_lz = wx, wy, wz

        # Rim points (4 cardinal + 4 diagonal)
        angles = (0.0, 1.5707963, 3.1415926, 4.7123890,
                  0.7853982, 2.3561945, 3.9269908, 5.4977871)
        for a in angles:
            ca = math.cos(a)
            sa = math.sin(a)
            for sign in (-1.0, 1.0):
                bx = r * ca
                by = sign * ht
                bz = r * sa
                wx, wy, wz = quat_rotate_vector(qw, qx, qy, qz, bx, by, bz)
                world_y = pos_y + wy
                if world_y < best_y:
                    best_y = world_y
                    best_lx, best_ly, best_lz = wx, wy, wz

        return best_y, best_lx, best_ly, best_lz

    @cuda.jit(device=True)
    def lowest_point_cube(qw, qx, qy, qz, pos_y):
        h = 0.15  # side/2
        best_y = 1e30
        best_lx = 0.0
        best_ly = 0.0
        best_lz = 0.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    bx = sx * h
                    by = sy * h
                    bz = sz * h
                    wx, wy, wz = quat_rotate_vector(
                        qw, qx, qy, qz, bx, by, bz)
                    world_y = pos_y + wy
                    if world_y < best_y:
                        best_y = world_y
                        best_lx, best_ly, best_lz = wx, wy, wz
        return best_y, best_lx, best_ly, best_lz

    @cuda.jit(device=True)
    def lowest_point_rod(qw, qx, qy, qz, pos_y):
        half = 0.5  # length/2
        rad = 0.02
        best_y = 1e30
        best_lx = 0.0
        best_ly = 0.0
        best_lz = 0.0
        for ex in (-1.0, 1.0):
            bx = ex * half
            for ey in (-1.0, 1.0):
                for ez in (-1.0, 1.0):
                    by = ey * rad
                    bz = ez * rad
                    wx, wy, wz = quat_rotate_vector(
                        qw, qx, qy, qz, bx, by, bz)
                    world_y = pos_y + wy
                    if world_y < best_y:
                        best_y = world_y
                        best_lx, best_ly, best_lz = wx, wy, wz
        return best_y, best_lx, best_ly, best_lz

    @cuda.jit(device=True)
    def lowest_point(shape_id, qw, qx, qy, qz, pos_y):
        if shape_id == SHAPE_COIN:
            return lowest_point_coin(qw, qx, qy, qz, pos_y)
        elif shape_id == SHAPE_CUBE:
            return lowest_point_cube(qw, qx, qy, qz, pos_y)
        else:
            return lowest_point_rod(qw, qx, qy, qz, pos_y)

    # ---------------------------------------------------------------
    # Floor constraint (normal impulse + friction + rolling resistance)
    # ---------------------------------------------------------------

    @cuda.jit(device=True)
    def cross3(ax, ay, az, bx, by, bz):
        return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

    @cuda.jit(device=True)
    def dot3(ax, ay, az, bx, by, bz):
        return ax*bx + ay*by + az*bz

    @cuda.jit(device=True)
    def floor_constraint(
        shape_id, mass,
        pos_x, pos_y, pos_z,
        px, py, pz,
        qw, qx, qy, qz,
        Lx, Ly, Lz,
        Ix, Iy, Iz,
        restitution, friction, rolling_resistance,
    ):
        """
        Enforce floor at y=0. Returns updated state tuple:
        (pos_x, pos_y, pos_z, px, py, pz, qw, qx, qy, qz, Lx, Ly, Lz)
        """
        world_y, lev_x, lev_y, lev_z = lowest_point(
            shape_id, qw, qx, qy, qz, pos_y)
        penetration = world_y
        in_contact = penetration < 0.0
        near_floor = penetration < 0.005

        if in_contact:
            pos_y -= penetration

            # Normal direction is (0, 1, 0)
            # Compute contact velocity
            safe_Ix = Ix if Ix > 1e-30 else 1e-30
            safe_Iy = Iy if Iy > 1e-30 else 1e-30
            safe_Iz = Iz if Iz > 1e-30 else 1e-30
            ox = Lx / safe_Ix
            oy = Ly / safe_Iy
            oz = Lz / safe_Iz

            # omega in world frame
            owx, owy, owz = quat_rotate_vector(qw, qx, qy, qz, ox, oy, oz)

            vx = px / mass
            vy = py / mass
            vz = pz / mass
            # v_contact = velocity + omega x lever
            cx, cy, cz = cross3(owx, owy, owz, lev_x, lev_y, lev_z)
            vc_x = vx + cx
            vc_y = vy + cy
            vc_z = vz + cz

            v_n = vc_y  # dot with (0,1,0)

            if v_n < -0.01:
                # Compute effective mass for normal impulse
                # r x n where n = (0,1,0)
                rcn_x, rcn_y, rcn_z = cross3(
                    lev_x, lev_y, lev_z, 0.0, 1.0, 0.0)

                # I_inv_world @ (r x n): transform to body, apply I_inv, transform back
                cw, ccx, ccy, ccz = quat_conjugate(qw, qx, qy, qz)
                bx2, by2, bz2 = quat_rotate_vector(
                    cw, ccx, ccy, ccz, rcn_x, rcn_y, rcn_z)
                ib_x = bx2 / safe_Ix
                ib_y = by2 / safe_Iy
                ib_z = bz2 / safe_Iz
                iw_x, iw_y, iw_z = quat_rotate_vector(
                    qw, qx, qy, qz, ib_x, ib_y, ib_z)

                # cross(I_inv_world @ (r x n), lever) . n
                t_x, t_y, t_z = cross3(iw_x, iw_y, iw_z,
                                        lev_x, lev_y, lev_z)
                rot_term = t_y  # dot with (0,1,0)
                inv_mass_eff = 1.0 / mass + rot_term

                e_eff = restitution if abs(v_n) > 0.1 else 0.0
                j_n = -(1.0 + e_eff) * v_n / inv_mass_eff

                # Apply normal impulse
                py += j_n  # j_n * n_y where n_y = 1

                # Torque impulse in body frame
                ti_x, ti_y, ti_z = cross3(lev_x, lev_y, lev_z,
                                           0.0, j_n, 0.0)
                tb_x, tb_y, tb_z = quat_rotate_vector(
                    cw, ccx, ccy, ccz, ti_x, ti_y, ti_z)
                Lx += tb_x
                Ly += tb_y
                Lz += tb_z

                # Coulomb friction
                if friction > 0.0:
                    safe_Ix2 = Ix if Ix > 1e-30 else 1e-30
                    safe_Iy2 = Iy if Iy > 1e-30 else 1e-30
                    safe_Iz2 = Iz if Iz > 1e-30 else 1e-30
                    ox2 = Lx / safe_Ix2
                    oy2 = Ly / safe_Iy2
                    oz2 = Lz / safe_Iz2
                    owx2, owy2, owz2 = quat_rotate_vector(
                        qw, qx, qy, qz, ox2, oy2, oz2)
                    vx2 = px / mass
                    vy2 = py / mass
                    vz2 = pz / mass
                    cx2, cy2, cz2 = cross3(
                        owx2, owy2, owz2, lev_x, lev_y, lev_z)
                    vc_x2 = vx2 + cx2
                    vc_y2 = vy2 + cy2
                    vc_z2 = vz2 + cz2
                    vn2 = vc_y2
                    vt_x = vc_x2
                    vt_y = vc_y2 - vn2
                    vt_z = vc_z2
                    vt_mag = math.sqrt(vt_x*vt_x + vt_y*vt_y + vt_z*vt_z)

                    if vt_mag > 1e-12:
                        inv_vt = 1.0 / vt_mag
                        th_x = vt_x * inv_vt
                        th_y = vt_y * inv_vt
                        th_z = vt_z * inv_vt

                        rct_x, rct_y, rct_z = cross3(
                            lev_x, lev_y, lev_z, th_x, th_y, th_z)
                        bt_x, bt_y, bt_z = quat_rotate_vector(
                            cw, ccx, ccy, ccz, rct_x, rct_y, rct_z)
                        ibt_x = bt_x / safe_Ix2
                        ibt_y = bt_y / safe_Iy2
                        ibt_z = bt_z / safe_Iz2
                        iwt_x, iwt_y, iwt_z = quat_rotate_vector(
                            qw, qx, qy, qz, ibt_x, ibt_y, ibt_z)
                        tt_x, tt_y, tt_z = cross3(
                            iwt_x, iwt_y, iwt_z,
                            lev_x, lev_y, lev_z)
                        rot_term_t = dot3(
                            tt_x, tt_y, tt_z, th_x, th_y, th_z)
                        inv_mass_eff_t = 1.0 / mass + rot_term_t
                        if inv_mass_eff_t < 1e-30:
                            inv_mass_eff_t = 1.0 / mass

                        j_t_desired = vt_mag / inv_mass_eff_t
                        j_t_max = friction * abs(j_n)
                        j_t = j_t_desired if j_t_desired < j_t_max else j_t_max

                        px -= j_t * th_x
                        py -= j_t * th_y
                        pz -= j_t * th_z
                        tf_x, tf_y, tf_z = cross3(
                            lev_x, lev_y, lev_z,
                            -j_t * th_x, -j_t * th_y, -j_t * th_z)
                        tfb_x, tfb_y, tfb_z = quat_rotate_vector(
                            cw, ccx, ccy, ccz, tf_x, tf_y, tf_z)
                        Lx += tfb_x
                        Ly += tfb_y
                        Lz += tfb_z

        # Rolling resistance and rest detection
        if near_floor and rolling_resistance > 0.0:
            min_h = get_min_rest_height(shape_id)
            excess = pos_y - min_h
            if excess < 0.0:
                excess = 0.0
            stability = 1.0 - excess / 0.05
            if stability < 0.0:
                stability = 0.0

            eff_rr = rolling_resistance * stability
            if eff_rr > 0.0:
                Lx *= (1.0 - eff_rr)
                Ly *= (1.0 - eff_rr)
                Lz *= (1.0 - eff_rr)

            at_rest = excess < 0.02
            ke_trans = (px*px + py*py + pz*pz) / (2.0 * mass)
            safe_Ix3 = Ix if Ix > 1e-30 else 1e-30
            safe_Iy3 = Iy if Iy > 1e-30 else 1e-30
            safe_Iz3 = Iz if Iz > 1e-30 else 1e-30
            ke_rot = 0.5 * (Lx*Lx/safe_Ix3 + Ly*Ly/safe_Iy3 + Lz*Lz/safe_Iz3)
            ke = ke_trans + ke_rot
            if ke < 0.005 and at_rest:
                px = 0.0
                py = 0.0
                pz = 0.0
                Lx = 0.0
                Ly = 0.0
                Lz = 0.0

        return pos_x, pos_y, pos_z, px, py, pz, qw, qx, qy, qz, Lx, Ly, Lz

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    @cuda.jit(device=True)
    def classify_coin_dev(qw, qx, qy, qz):
        _, wy, _ = quat_rotate_vector(qw, qx, qy, qz, 0.0, 1.0, 0.0)
        if wy > 0.1:
            return 1    # heads
        if wy < -0.1:
            return -1   # tails
        return 0        # edge

    @cuda.jit(device=True)
    def classify_cube_dev(qw, qx, qy, qz):
        """Return 0-5 for which body-frame axis points most downward."""
        r10, r11, r12 = quat_to_rotmat_row(qw, qx, qy, qz, 1)
        # y-components of each of 6 body axes:
        # +x axis -> R[1,0], -x -> -R[1,0], +y -> R[1,1], -y -> -R[1,1],
        # +z -> R[1,2], -z -> -R[1,2]
        vals_0 = r10     # +x
        vals_1 = -r10    # -x
        vals_2 = r11     # +y
        vals_3 = -r11    # -y
        vals_4 = r12     # +z
        vals_5 = -r12    # -z
        min_val = vals_0
        min_idx = 0
        if vals_1 < min_val:
            min_val = vals_1
            min_idx = 1
        if vals_2 < min_val:
            min_val = vals_2
            min_idx = 2
        if vals_3 < min_val:
            min_val = vals_3
            min_idx = 3
        if vals_4 < min_val:
            min_val = vals_4
            min_idx = 4
        if vals_5 < min_val:
            min_idx = 5
        return min_idx

    @cuda.jit(device=True)
    def classify_rod_dev(qw, qx, qy, qz):
        _, wy, _ = quat_rotate_vector(qw, qx, qy, qz, 1.0, 0.0, 0.0)
        if wy < -0.1:
            return 1    # +x end down
        if wy > 0.1:
            return -1   # -x end down
        return 0        # flat

    @cuda.jit(device=True)
    def classify_dev(shape_id, qw, qx, qy, qz):
        if shape_id == SHAPE_COIN:
            return classify_coin_dev(qw, qx, qy, qz)
        elif shape_id == SHAPE_CUBE:
            return classify_cube_dev(qw, qx, qy, qz)
        else:
            return classify_rod_dev(qw, qx, qy, qz)

    # ===============================================================
    # Main kernel
    # ===============================================================

    @cuda.jit
    def drop_kernel(
        heights, angles,
        axis_x, axis_y, axis_z,
        shape_id,
        dt, n_max_steps,
        restitution, friction, rolling_resistance,
        g,
        results,
    ):
        """One CUDA thread per (height, angle) drop simulation."""
        i, j = cuda.grid(2)
        if i >= heights.shape[0] or j >= angles.shape[0]:
            return

        h = heights[i]
        angle = angles[j]

        # Initial orientation from axis-angle
        qw, qx, qy, qz = quat_from_axis_angle(axis_x, axis_y, axis_z, angle)

        # Initial state
        pos_x = 0.0
        pos_y = h
        pos_z = 0.0
        px = 0.0
        py = 0.0
        pz = 0.0
        Lx = 0.0
        Ly = 0.0
        Lz = 0.0

        Ix, Iy, Iz = get_inertia(shape_id)
        mass = 1.0

        # Settle detection
        settle_h = 0.2
        if shape_id == SHAPE_CUBE:
            settle_h = 0.35
        elif shape_id == SHAPE_ROD:
            settle_h = 0.55

        settled_count = 0

        for step in range(n_max_steps):
            # --- First half-kick (gravity only: Fy = -m*g) ---
            half_dt = dt * 0.5
            py -= mass * g * half_dt

            # --- Drift ---
            inv_m = 1.0 / mass
            pos_x += px * inv_m * dt
            pos_y += py * inv_m * dt
            pos_z += pz * inv_m * dt

            # Quaternion drift
            safe_Ix = Ix if Ix > 1e-30 else 1e-30
            safe_Iy = Iy if Iy > 1e-30 else 1e-30
            safe_Iz = Iz if Iz > 1e-30 else 1e-30
            ox = Lx / safe_Ix
            oy = Ly / safe_Iy
            oz = Lz / safe_Iz
            dqw, dqx, dqy, dqz = quat_exp_map(ox, oy, oz, dt)
            qw, qx, qy, qz = quat_multiply(dqw, dqx, dqy, dqz,
                                             qw, qx, qy, qz)
            qw, qx, qy, qz = quat_normalize(qw, qx, qy, qz)

            # --- Constraint enforcement ---
            (pos_x, pos_y, pos_z,
             px, py, pz,
             qw, qx, qy, qz,
             Lx, Ly, Lz) = floor_constraint(
                shape_id, mass,
                pos_x, pos_y, pos_z,
                px, py, pz,
                qw, qx, qy, qz,
                Lx, Ly, Lz,
                Ix, Iy, Iz,
                restitution, friction, rolling_resistance)

            # --- Second half-kick ---
            py -= mass * g * half_dt

            # --- Settle detection ---
            ke_trans = (px*px + py*py + pz*pz) * 0.5 * inv_m
            ke_rot = 0.5 * (Lx*Lx/safe_Ix + Ly*Ly/safe_Iy + Lz*Lz/safe_Iz)
            ke = ke_trans + ke_rot

            if ke < 1e-6 and pos_y < settle_h:
                settled_count += 1
                if settled_count > 200:
                    break
            else:
                settled_count = 0

        results[i, j] = classify_dev(shape_id, qw, qx, qy, qz)


# ====================================================================
# Host-side interface
# ====================================================================

def sweep_drop_gpu(shape, heights, angles, tilt_axis="x",
                   dt=0.001, restitution=0.6, friction=0.5,
                   rolling_resistance=0.05, g=9.81, duration=None):
    """
    GPU-accelerated parameter sweep.

    Parameters
    ----------
    shape : str
        "coin", "cube", or "rod".
    heights, angles : array-like
        1-D arrays of drop heights and tilt angles (radians).
    tilt_axis : str
        "x", "y", or "z".
    dt : float
        Integration timestep.
    restitution, friction, rolling_resistance : float
        Floor constraint physics parameters.
    g : float
        Gravitational acceleration.
    duration : float or None
        Max simulation time (auto-calculated if None).

    Returns
    -------
    results : ndarray of int, shape (len(heights), len(angles))
    """
    if not HAS_CUDA:
        raise RuntimeError(
            "CUDA is not available. Install numba with CUDA toolkit:\n"
            "  pip install numba nvidia-cuda-nvcc-cu12==12.4.* "
            "nvidia-cuda-runtime-cu12==12.4.*\n"
            "and ensure LD_LIBRARY_PATH includes the CUDA libraries."
        )

    heights = np.asarray(heights, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    nh, na = len(heights), len(angles)

    shape_id = _SHAPE_NAME_TO_ID[shape]

    axis_map = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    ax, ay, az = axis_map[tilt_axis]

    if duration is None:
        max_h = float(heights.max())
        t_fall = math.sqrt(2.0 * max_h / g)
        duration = max(t_fall * 8, 2.0)

    n_max_steps = int(math.ceil(duration / dt))

    d_heights = cuda.to_device(heights)
    d_angles = cuda.to_device(angles)
    d_results = cuda.device_array((nh, na), dtype=np.int32)

    threads_per_block = (16, 16)
    blocks_x = (nh + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (na + threads_per_block[1] - 1) // threads_per_block[1]
    blocks = (blocks_x, blocks_y)

    drop_kernel[blocks, threads_per_block](
        d_heights, d_angles,
        ax, ay, az,
        shape_id,
        dt, n_max_steps,
        restitution, friction, rolling_resistance,
        g,
        d_results,
    )

    cuda.synchronize()
    results = d_results.copy_to_host()
    return results


def gpu_info():
    """Return a dict of GPU information, or None if unavailable."""
    if not HAS_CUDA:
        return None
    dev = cuda.get_current_device()
    return {
        "name": dev.name.decode() if isinstance(dev.name, bytes) else dev.name,
        "compute_capability": dev.compute_capability,
        "max_threads_per_block": dev.MAX_THREADS_PER_BLOCK,
        "warp_size": dev.WARP_SIZE,
    }
