#!/usr/bin/env python
"""
Drop-coin experiment — sweep height x tilt-angle, classify
heads/tails/edge, and show a progressive outcome map.

Usage:
    python experiments/drop_coin.py              # defaults
    python experiments/drop_coin.py --workers 4  # limit CPU cores
    python experiments/drop_coin.py --axis z     # tilt about z-axis
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
    sweep_drop, plot_drop_map, _results_to_rgb, _SHAPE_PALETTE,
)


def main():
    parser = argparse.ArgumentParser(description="Coin drop outcome map")
    parser.add_argument("--nh", type=int, default=40, help="height grid points")
    parser.add_argument("--na", type=int, default=60, help="angle grid points")
    parser.add_argument("--hmin", type=float, default=0.1)
    parser.add_argument("--hmax", type=float, default=3.0)
    parser.add_argument("--axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 4
    total = args.nh * args.na
    print(f"Coin drop experiment")
    print(f"  grid:    {args.nh} heights x {args.na} angles = {total} simulations")
    print(f"  heights: {args.hmin} – {args.hmax} m")
    print(f"  angles:  0 – 360° about {args.axis}-axis")
    print(f"  workers: {workers}")
    print()

    heights = np.linspace(args.hmin, args.hmax, args.nh)
    angles = np.linspace(0, 2 * np.pi, args.na, endpoint=False)

    results = np.full((args.nh, args.na), np.nan)
    colors, _ = _SHAPE_PALETTE["coin"]

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
    sweep_drop("coin", heights, angles, tilt_axis=args.axis,
               workers=workers, callback=on_result)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({total/elapsed:.0f} sims/sec)")
    print("Showing plot...")

    fig, ax, img = plot_drop_map(heights, angles, results.astype(int),
                                 "coin", args.axis)
    ax.set_title(f"coin drop outcome map — tilt about {args.axis}-axis")
    plt.show()


if __name__ == "__main__":
    main()
