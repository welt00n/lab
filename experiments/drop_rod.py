#!/usr/bin/env python
"""
Drop-rod experiment — sweep height x tilt-angle, classify
which end lands down (or flat), and show the outcome map.

Usage:
    python experiments/drop_rod.py                # CPU, batch mode
    python experiments/drop_rod.py --gpu          # GPU (CUDA)
    python experiments/drop_rod.py --live         # live 3-panel dashboard
    python experiments/drop_rod.py --live --gpu   # live + GPU sweep
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from lab.experiments.drop_experiment import (
    sweep_drop, plot_drop_map, _SHAPE_PALETTE,
)

SHAPE = "rod"


def _run_batch(args):
    workers = args.workers or os.cpu_count() or 4
    total = args.nh * args.na
    heights = np.linspace(args.hmin, args.hmax, args.nh)
    angles = np.linspace(0, 2 * np.pi, args.na, endpoint=False)
    results = np.full((args.nh, args.na), np.nan)

    print(f"{SHAPE.title()} drop experiment (CPU batch)")
    print(f"  grid:    {args.nh} x {args.na} = {total} simulations")
    print(f"  heights: {args.hmin} – {args.hmax} m")
    print(f"  tilt:    0–360° about {args.axis}-axis")
    print(f"  workers: {workers}")
    print()

    t0 = time.time()

    def on_result(i, j, result, done, total):
        results[i, j] = result
        if done % max(1, total // 20) == 0 or done == total:
            pct = 100 * done / total
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"  {pct:5.1f}%  ({done}/{total})  "
                  f"elapsed {elapsed:.1f}s  eta {eta:.1f}s", flush=True)

    print("Running simulations...")
    sweep_drop(SHAPE, heights, angles, tilt_axis=args.axis,
               workers=workers, callback=on_result)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({total / max(elapsed, 1e-9):.0f} sims/sec)")

    fig, ax, img = plot_drop_map(heights, angles, results.astype(int),
                                 SHAPE, args.axis)
    ax.set_title(f"{SHAPE} drop outcome — tilt about {args.axis}-axis")
    plt.show()


def _run_gpu(args):
    from lab.experiments.drop_gpu import sweep_drop_gpu, gpu_info

    total = args.nh * args.na
    heights = np.linspace(args.hmin, args.hmax, args.nh)
    angles = np.linspace(0, 2 * np.pi, args.na, endpoint=False)

    info = gpu_info()
    print(f"{SHAPE.title()} drop experiment (GPU)")
    print(f"  device:  {info['name']}")
    print(f"  grid:    {args.nh} x {args.na} = {total} simulations")
    print(f"  heights: {args.hmin} – {args.hmax} m")
    print(f"  tilt:    0–360° about {args.axis}-axis")
    print()

    t0 = time.time()
    print("Running kernel...")
    results = sweep_drop_gpu(SHAPE, heights, angles, tilt_axis=args.axis)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.3f}s ({total / max(elapsed, 1e-9):.0f} sims/sec)")

    fig, ax, img = plot_drop_map(heights, angles, results, SHAPE, args.axis)
    ax.set_title(f"{SHAPE} drop outcome (GPU) — tilt about {args.axis}-axis")
    plt.show()


def _run_live(args):
    from lab.experiments.live_dashboard import run_live_dashboard

    heights = np.linspace(args.hmin, args.hmax, args.nh)
    angles = np.linspace(0, 2 * np.pi, args.na, endpoint=False)

    print(f"{SHAPE.title()} drop experiment (live dashboard)")
    print(f"  grid:    {args.nh} x {args.na} = {args.nh * args.na} simulations")
    print(f"  heights: {args.hmin} – {args.hmax} m")
    print(f"  tilt:    0–360° about {args.axis}-axis")
    print(f"  mode:    {'GPU' if args.gpu else 'CPU'}")
    print()

    run_live_dashboard(SHAPE, heights, angles, tilt_axis=args.axis,
                       workers=args.workers, gpu=args.gpu)


def main():
    parser = argparse.ArgumentParser(description=f"{SHAPE.title()} drop outcome map")
    parser.add_argument("--nh", type=int, default=40, help="height grid points")
    parser.add_argument("--na", type=int, default=60, help="angle grid points")
    parser.add_argument("--hmin", type=float, default=0.1)
    parser.add_argument("--hmax", type=float, default=5.0)
    parser.add_argument("--axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="use GPU for sweep")
    parser.add_argument("--live", action="store_true",
                        help="live 3-panel dashboard with 3D animation")
    args = parser.parse_args()

    if args.live:
        _run_live(args)
    elif args.gpu:
        _run_gpu(args)
    else:
        _run_batch(args)


if __name__ == "__main__":
    main()
