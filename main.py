"""
lab — experiment launcher.

Usage:
    python main.py coin                    # CPU batch sweep
    python main.py coin --gpu              # GPU (CUDA) sweep
    python main.py coin --live             # live dashboard
    python main.py cube --nh 20 --na 30    # custom grid size
    python main.py                         # show help
"""

import sys
import os
import argparse
import time

import numpy as np


def _setup_nvidia_libs():
    """Configure env vars so numba finds pip-installed NVIDIA CUDA libs."""
    import glob, sysconfig
    sp = sysconfig.get_path("purelib")
    nv = os.path.join(sp, "nvidia")
    if not os.path.isdir(nv):
        return

    cuda_nvcc = os.path.join(nv, "cuda_nvcc")
    if os.path.isdir(cuda_nvcc) and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = cuda_nvcc

    lib_dirs = sorted(glob.glob(os.path.join(nv, "*", "lib*")))
    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new = ":".join(d for d in lib_dirs if d not in existing)
        if new:
            os.environ["LD_LIBRARY_PATH"] = (new + ":" + existing).rstrip(":")


_setup_nvidia_libs()


EXPERIMENTS = {
    "coin": "lab.experiments.coin.CoinDrop",
    "cube": "lab.experiments.cube.CubeDrop",
}


def _load_experiment(name):
    """Import and instantiate the named experiment class."""
    dotted = EXPERIMENTS[name]
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def _make_output_dir(shape, mode):
    """Create a dated output folder under results/."""
    from pathlib import Path
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{mode}" if mode else ""
    path = Path("results") / f"{stamp}_{shape}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_parameters(output_dir, args, elapsed):
    """Write run parameters to JSON for reproducibility."""
    import json
    from pathlib import Path
    params = {
        "experiment": args.experiment,
        "nh": args.nh,
        "na": args.na,
        "hmin": args.hmin,
        "hmax": args.hmax,
        "axis": args.axis,
        "mode": "gpu" if args.gpu else ("live" if args.live else "batch"),
        "save_video": args.save_video,
        "elapsed_seconds": round(elapsed, 2),
    }
    (Path(output_dir) / "parameters.json").write_text(
        json.dumps(params, indent=2) + "\n")


def welcome():
    BOLD, DIM, CYAN = "\033[1m", "\033[2m", "\033[36m"
    YELLOW, GREEN, RESET = "\033[33m", "\033[32m", "\033[0m"

    print(f"""
{BOLD}{CYAN}\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569
\u2551                                                              \u2551
\u2551   \u269b  lab \u2014 Hamiltonian Physics Lab                           \u2551
\u2551                                                              \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c{RESET}

{BOLD}{YELLOW}  Experiments{RESET}
{DIM}  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{RESET}
  {GREEN}python main.py coin{RESET}              Coin drop (CPU batch)
  {GREEN}python main.py coin --live{RESET}       Coin drop (live dashboard)
  {GREEN}python main.py coin --gpu{RESET}        Coin drop (GPU CUDA)
  {GREEN}python main.py cube{RESET}              Cube drop (CPU batch)
  {GREEN}python main.py cube --live{RESET}       Cube drop (live dashboard)
  {GREEN}python main.py cube --gpu{RESET}        Cube drop (GPU CUDA)

{BOLD}{YELLOW}  Options{RESET}
{DIM}  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{RESET}
  --nh N       Height grid points (default: 40)
  --na N       Angle grid points (default: 60)
  --hmin F     Min height in metres (default: 0.1)
  --hmax F     Max height in metres (default: 5.0)
  --axis X     Tilt axis: x, y, or z (default: x)
  --live       Live 3D dashboard
  --gpu        Use CUDA GPU for sweep
  --save-video Save animated replay as MP4

{BOLD}{YELLOW}  Tests{RESET}
{DIM}  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{RESET}
  {GREEN}python -m pytest tests/ -v{RESET}
""")


def main():
    parser = argparse.ArgumentParser(
        description="lab \u2014 Hamiltonian Physics Lab experiment launcher")
    parser.add_argument("experiment", nargs="?", default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help="experiment to run")
    parser.add_argument("--nh", type=int, default=40,
                        help="height grid points")
    parser.add_argument("--na", type=int, default=60,
                        help="angle grid points")
    parser.add_argument("--hmin", type=float, default=0.1,
                        help="min drop height (m)")
    parser.add_argument("--hmax", type=float, default=5.0,
                        help="max drop height (m)")
    parser.add_argument("--axis", default="x", choices=["x", "y", "z"],
                        help="tilt axis")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU (CUDA) for parameter sweep")
    parser.add_argument("--live", action="store_true",
                        help="live 3D dashboard with animated physics")
    parser.add_argument("--save-video", action="store_true",
                        help="save animated replay as MP4 to results folder")

    args = parser.parse_args()

    if args.experiment is None:
        welcome()
        return

    exp = _load_experiment(args.experiment)
    heights, angles = exp.build_grid(args.nh, args.na, args.hmin, args.hmax)
    total = args.nh * args.na

    if args.live:
        mode = "live"
        print(f"{exp.shape.title()} drop experiment (live dashboard)")
        print(f"  grid:    {args.nh} x {args.na} = {total} simulations")
        print(f"  heights: {args.hmin} \u2013 {args.hmax} m")
        print(f"  tilt:    about {args.axis}-axis")
        print()
        exp.run_live(heights, angles, tilt_axis=args.axis)

    elif args.gpu:
        mode = "gpu"
        from lab.experiments.drop_gpu import gpu_info
        info = gpu_info()
        if info is None:
            print("ERROR: CUDA is not available.")
            sys.exit(1)
        print(f"{exp.shape.title()} drop experiment (GPU)")
        print(f"  device:  {info['name']}")
        print(f"  grid:    {args.nh} x {args.na} = {total} simulations")
        print(f"  heights: {args.hmin} \u2013 {args.hmax} m")
        print()
        t0 = time.time()
        results = exp.sweep_gpu(heights, angles, tilt_axis=args.axis)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.3f}s "
              f"({total / max(elapsed, 1e-9):.0f} sims/sec)")

        output_dir = _make_output_dir(exp.shape, mode)
        _save_parameters(output_dir, args, elapsed)
        exp.show_results(heights, angles, results,
                         tilt_axis=args.axis, mode="GPU",
                         output_dir=str(output_dir))
        if args.save_video:
            exp.save_video(heights, angles, results,
                           tilt_axis=args.axis,
                           output_dir=str(output_dir))

    else:
        mode = "batch"
        print(f"{exp.shape.title()} drop experiment (CPU batch)")
        print(f"  grid:    {args.nh} x {args.na} = {total} simulations")
        print(f"  heights: {args.hmin} \u2013 {args.hmax} m")
        print(f"  tilt:    about {args.axis}-axis")
        print()
        t0 = time.time()
        results = exp.sweep(heights, angles, tilt_axis=args.axis)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s "
              f"({total / max(elapsed, 1e-9):.0f} sims/sec)")

        output_dir = _make_output_dir(exp.shape, mode)
        _save_parameters(output_dir, args, elapsed)
        exp.show_results(heights, angles, results,
                         tilt_axis=args.axis, mode="CPU",
                         output_dir=str(output_dir))
        if args.save_video:
            exp.save_video(heights, angles, results,
                           tilt_axis=args.axis,
                           output_dir=str(output_dir))


if __name__ == "__main__":
    main()
