"""
DropExperiment base class — the central experiment abstraction.

Subclasses declare shape-specific config (colors, labels, mesh, etc.)
and inherit sweep/live/GPU execution plus visualization for free.
"""

from __future__ import annotations

import math
import time
import numpy as np

from lab.core.rigid_body_jit import (
    step_bodies, classify, warmup,
    quat_from_axis_angle, SHAPE_NAME_TO_ID,
)

_AXIS_MAP = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


class DropExperiment:
    """
    Base class for rigid-body drop experiments.

    Subclasses set declarative attributes only — no methods needed.
    All execution modes (batch, GPU, live) and visualization are
    handled here using generic primitives.
    """

    # --- Subclass provides these ---
    shape: str
    shape_id: int
    angle_range: tuple[float, float]
    colors: dict[int, str]
    labels: dict[int, str]
    settle_height: float
    body_color: str
    mesh: str

    def build_grid(self, nh, na, hmin, hmax):
        """Return (heights, angles) arrays for the parameter sweep."""
        heights = np.linspace(hmin, hmax, nh)
        angles = np.linspace(
            self.angle_range[0], self.angle_range[1], na, endpoint=False
        )
        return heights, angles

    def sweep(self, heights, angles, tilt_axis="x",
              dt=0.0005, g=9.81, rest=0.6, fric=0.5, rr=0.05):
        """
        Batch CPU sweep — run all simulations to completion.

        Returns results[nh, na] integer grid.
        """
        nh, na = len(heights), len(angles)
        N = nh * na
        ax_f, ay_f, az_f = _AXIS_MAP[tilt_axis]

        print(f"  Compiling physics (first run only)...", end=" ", flush=True)
        warmup(self.shape_id)
        print("done.", flush=True)

        pos = np.zeros((N, 3), dtype=np.float64)
        mom = np.zeros((N, 3), dtype=np.float64)
        ori = np.zeros((N, 4), dtype=np.float64)
        amom = np.zeros((N, 3), dtype=np.float64)
        alive = np.ones(N, dtype=np.bool_)
        sc = np.zeros(N, dtype=np.int64)
        alive_idx = np.arange(N, dtype=np.int64)

        grid_ij = np.zeros((N, 2), dtype=np.int64)
        idx = 0
        for i, h in enumerate(heights):
            for j, a in enumerate(angles):
                pos[idx] = [0.0, float(h), 0.0]
                w, x, y, z = quat_from_axis_angle(ax_f, ay_f, az_f, float(a))
                ori[idx] = [w, x, y, z]
                grid_ij[idx] = [i, j]
                idx += 1

        n_alive = N
        max_iters = 200_000
        batch_steps = 500

        t0 = time.time()
        iters = 0
        while n_alive > 0 and iters < max_iters:
            _, n_alive = step_bodies(
                pos, mom, ori, amom, alive, sc, alive_idx, n_alive,
                self.shape_id, dt, g, rest, fric, rr,
                self.settle_height, batch_steps,
            )
            iters += 1

        elapsed = time.time() - t0
        print(f"  Batch done in {elapsed:.1f}s")

        results = np.full((nh, na), -99, dtype=np.int64)
        for k in range(N):
            i2, j2 = int(grid_ij[k, 0]), int(grid_ij[k, 1])
            results[i2, j2] = classify(
                self.shape_id,
                ori[k, 0], ori[k, 1], ori[k, 2], ori[k, 3],
            )

        return results

    def sweep_gpu(self, heights, angles, tilt_axis="x", **kwargs):
        """GPU sweep — delegates to the CUDA kernel wrapper."""
        from lab.experiments.drop_gpu import sweep_drop_gpu
        return sweep_drop_gpu(self.shape, heights, angles,
                              tilt_axis=tilt_axis, **kwargs)

    def run_live(self, heights, angles, tilt_axis="x",
                 workers=None, gpu=False):
        """Launch the live dashboard."""
        from lab.visualization.dashboard import run_live
        run_live(self, heights, angles, tilt_axis=tilt_axis)

    def show_results(self, heights, angles, results, tilt_axis="x",
                     mode="", output_dir=None):
        """Static display of completed results."""
        from lab.visualization.dashboard import show_results
        show_results(self, heights, angles, results,
                     tilt_axis=tilt_axis, mode=mode, output_dir=output_dir)

    def run_replay(self, heights, angles, results, tilt_axis="x"):
        """Animated replay of pre-computed results with 3D scene."""
        from lab.visualization.dashboard import run_replay
        run_replay(self, heights, angles, results, tilt_axis=tilt_axis)

    def save_video(self, heights, angles, results,
                   tilt_axis="x", output_dir=None, fast=False):
        """Save an animated replay as MP4 to *output_dir*."""
        from lab.visualization.dashboard import save_video
        save_video(self, heights, angles, results,
                   tilt_axis=tilt_axis, output_dir=output_dir, fast=fast)

    @staticmethod
    def default_args():
        """CLI defaults shared across all drop experiments."""
        return dict(nh=40, na=60, hmin=0.1, hmax=5.0, axis="x")
