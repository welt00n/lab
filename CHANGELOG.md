# Changelog

All notable changes to **lab** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.7.0] — 2026-03-02

### Added
- **Experiment framework.** New `DropExperiment` base class
  (`lab/experiments/base.py`) with declarative subclasses `CoinDrop`
  (`lab/experiments/coin.py`) and `CubeDrop` (`lab/experiments/cube.py`).
  Experiments define only shape-specific config (colors, labels, mesh,
  angle range, settle height); all physics and visualization are inherited.
- **Single physics core** (`lab/core/rigid_body_jit.py`). All `@njit`
  physics — quaternion math, lowest-point calculation, floor constraint,
  batch stepper (`step_bodies`), classification — consolidated into one
  494-line module. Constants defined once; CPU batch, live, and GPU paths
  all import from here.
- **Generic visualization primitives**:
  - `lab/visualization/sweep_grid.py` — categorical 2D color grid with
    live `update()`. Replaces 3 prior implementations.
  - `lab/visualization/category_histogram.py` — discrete bar chart with
    live `update()`. Replaces 2 prior implementations.
  - `lab/visualization/body_scene.py` — 3D rigid-body renderer (PyVista
    or matplotlib fallback) with `update()`/`mark_settled()`.
  - `lab/visualization/playback_controls.py` — reusable pause/speed/step
    widgets. Replaces 4 prior copies.
  - `lab/visualization/dashboard.py` — compositor wiring primitives to
    experiment config for `show_results`, `run_live`, `run_replay`, and
    `save_video` modes.
- **Video export.** `--save-video` CLI flag renders an animated MP4 replay
  (3D scene + outcome map + histogram) via `FuncAnimation.save()`. Falls
  back to GIF via Pillow if FFmpeg is unavailable. Uses bundled FFmpeg
  from `imageio-ffmpeg` when system FFmpeg is not installed.
- **Full 360° coin sweep.** Coin angle range changed from 0–π to 0–2π.
- **Auto-save results.** Every run saves outcome map PNG, histogram PNG,
  and `parameters.json` to a dated folder under `results/`.
- **Makefile** with targets: `install`, `coin`, `cube`, `demo-coin`,
  `demo-cube`, `test`, `test-fast`, `clean`, `help`. Exports CUDA
  environment variables for GPU runs.
- **CUDA auto-discovery.** `_setup_nvidia_libs()` in `main.py`
  configures `CUDA_HOME` and `LD_LIBRARY_PATH` from pip-installed NVIDIA
  packages, enabling Numba CUDA without a system-wide CUDA toolkit.
- **Comprehensive test suite** — 103 new test methods across 4 files:
  - `tests/test_rigid_body_jit.py` (47 tests) — quaternion math,
    lowest-point, classify, get_mass/get_inertia, step_bodies, warmup.
  - `tests/test_experiments.py` (24 tests) — subclass config, build_grid,
    sweep output, default_args.
  - `tests/test_visualization.py` (11 tests) — sweep_grid and
    category_histogram data transforms and artist creation.
  - `tests/test_cli.py` (19 tests) — experiment loading, output dir
    creation, argument parsing, NVIDIA setup safety.

### Changed
- **`main.py` rewritten** as a thin experiment launcher (~120 lines).
  Discovers experiments from a registry, parses shared CLI args
  (`--nh`, `--na`, `--hmin`, `--hmax`, `--axis`, `--gpu`, `--live`,
  `--save-video`), dispatches to the appropriate experiment method.
  All 9 inline demo functions removed.
- **`lab/experiments/drop_gpu.py`** now imports constants from the JIT
  core instead of duplicating them. Fixed namespace package path
  resolution (`__path__` instead of `__file__`).
- **`experiments/drop_coin.py`** and **`experiments/drop_cube.py`**
  converted to thin redirect scripts that invoke `main.py`.
- **`completions.bash`** rewritten for the new `main.py` CLI.
- **`requirements.txt`** updated with `imageio-ffmpeg>=0.5` and
  `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvcc-cu12`,
  `nvidia-cuda-nvrtc-cu12` for pip-based CUDA support.
- **Documentation** — all `docs/*.md` files updated to reflect the new
  architecture: file paths, function names, CLI examples, and code
  snippets corrected throughout.

### Removed
- `lab/experiments/live_dashboard.py` (1054 lines) — physics extracted
  to JIT core, UI replaced by `dashboard.py` + visualization primitives.
- `lab/experiments/drop_experiment.py` (830 lines) — absorbed into
  `base.py` and visualization primitives.
- `lab/visualization/pyvista_scene.py` (264 lines) — absorbed into
  `body_scene.py`.
- `lab/lab/__init__.py` — empty nested package directory.

---

## [0.6.0] — 2026-02-23

### Added
- **Realistic physical parameters.** Coin modelled on a US quarter (24.26 mm
  diameter, 1.75 mm thick, 5.67 g cupronickel). Cube modelled on a standard
  16 mm casino die (8 g acrylic). Physics dimensions decoupled from visual
  mesh — display meshes are ~12× larger for visibility, while all physics
  calculations use real-world geometry.
- **Mass-scaled thresholds.** All energy-based thresholds (settle KE, damping
  zones, snap-to-zero) now scale with `ke_scale = m * g * char_size`, the PE
  of lifting the object by its own size. Prevents threshold mismatches when
  switching between large/small objects.
- **PyVista 3D rendering.** Live dashboard and batch result viewer now use
  PyVista (VTK/OpenGL) for GPU-accelerated 3D scenes showing all bodies
  falling simultaneously. Falls back to matplotlib `Poly3DCollection` when
  PyVista is not installed.
  - `lab/visualization/pyvista_scene.py` — `DropScene` class managing N
    actors on a floor plane with coordinate transform (physics y-up → display
    z-up).
  - All N bodies rendered (no 16-sample cap); floor plane auto-sized to fit.
- **Wall-clock synchronised playback.** The "Play" button in the batch result
  viewer advances simulation time in sync with real time via
  `time.monotonic()`. Pause/resume and speed controls on both live and batch
  viewers.
- **New documentation**:
  - `docs/README.md` — documentation index with cross-link diagram and entry
    points by interest.
  - `docs/contact_model.md` — full floor constraint deep dive (~450 lines):
    penetration detection, normal impulse derivation, Coulomb friction, rolling
    resistance, damping zones, settle detection, contact resolution flowchart,
    code mapping across OOP/JIT/CUDA.
  - `docs/realistic_parameters.md` — real-world constants, inertia tensor
    derivations, scale invariance, dimensional analysis for thresholds,
    timestep justification, visual vs physical mesh.
  - `docs/drop_experiment.md` — experiment pipeline anatomy: parameter space,
    three execution paths (CPU/GPU/live), classification, outcome map
    interpretation, visualization architecture.

### Changed
- **Timestep halved** from `dt = 0.001` to `dt = 0.0005` for numerical
  stability with realistic (small, light) objects. Worst-case angular velocity
  after edge impact is ~2000 rad/s for a US quarter, requiring the smaller dt.
- **Rod experiment removed.** Only coin and cube experiments remain.
  `experiments/drop_rod.py` deleted; all rod references removed from docs.
- **Documentation deep refactor** — all 9 existing docs updated:
  - `physics.md` — expanded non-conservative systems section, fixed energy
    equation, updated parameter tables to real values, added units convention,
    extracted contact physics to `contact_model.md`.
  - `integration.md` — recalculated half-kick residual for real mass/dt, added
    Lie-group / symplecticity discussion for quaternion rotations, added
    timestep selection guidance, added dimensionally-scaled settle threshold.
  - `numerical_methods.md` — updated threshold table with `ke_scale` column,
    fixed RK45 description (embedded pair, not two-half-step), added
    dimensional analysis section, added quaternion drift analysis, updated
    damping values and force-settle timeout.
  - `rotations.md` — deepened SU(2) double-cover explanation, added quaternion
    kinematics section (dq/dt = ½ω·q), added numerical drift and
    renormalisation section.
  - `architecture.md` — removed rod from project tree, added
    three-implementations comparison (OOP/JIT/CUDA), added constants
    synchronisation note, added live dashboard architecture section, updated
    test count.
  - `gpu.md` — updated settle thresholds to mass-scaled values, added
    troubleshooting section (NvvmSupportError, compute capability, OOM),
    added numerical reproducibility cross-ref, labeled hardware specs as
    example, expanded performance section for halved dt.
  - `parallelism.md` — condensed Flynn's taxonomy (~55→20 lines), added
    load-balancing note, added JIT compilation overhead note, updated timing
    claims for halved dt.
  - `chaos.md` — updated dt reference, added double pendulum KE equation,
    added numerical-vs-physical chaos section, added fractal basin dimension
    discussion, added drop experiment connection note.

---

## [0.5.0] — 2026-02-26

### Added
- **JIT-compiled live dashboard** (`lab/experiments/live_dashboard.py`).
  The `--live` flag now opens a three-panel animation where all grid points
  drop simultaneously as a 3D scatter, the outcome map fills in real time,
  and a histogram tracks the distribution.
  - Physics reimplemented as `@njit(cache=True)` scalar functions — same
    leapfrog + floor-constraint math as the GPU kernel, compiled to native
    x86-64 via Numba.
  - ~300× faster than Python `World` objects: 2400 bodies × 10 steps in
    ~8 ms, enabling interactive frame rates.
  - Compiled binaries cached to disk; first run compiles in ~2 s, subsequent
    runs start instantly.
  - Single-thread architecture: physics + rendering on the main thread inside
    `FuncAnimation.update()`.  No background workers, no queues, no deadlocks.
  - `scatter3D` rendering — one draw call for all N objects, updated via
    `_offsets3d` and `set_facecolors`.
  - Vectorised colour/position updates using numpy indexing.
- **Settle-detection fix**: the energy check now runs *between* the floor
  constraint and the second leapfrog half-kick, avoiding the operator-
  splitting artefact where the half-kick reinjects residual momentum into
  a body that is physically at rest.
- **Force-settle timeout**: bodies that remain below `settle_h` for 5000
  cumulative steps are declared settled regardless of residual KE, preventing
  infinite rocking in quasi-periodic edge cases.
- **Near-floor damping zone** widened from 0.005 m to 0.05 m, with
  progressive linear velocity damping (0.998× per step) and a more aggressive
  KE snap-to-zero threshold (0.1 J within 0.05 m of rest height).

### Changed
- **3D visualisation axes remapped for physical clarity**: z-axis now shows
  true height (metres), x-axis shows the tilt-angle parameter offset, y-axis
  shows the initial drop-height parameter offset.  Camera view set to
  `elev=20, azim=-50` for a natural perspective.  Tick formatting cleaned up
  with `MaxNLocator(5)` and smaller fonts.
- Default `--hmax` raised from 3.0 m to 5.0 m in all three experiment
  scripts to better illustrate chaotic behaviour at higher drop heights.
- **`completions.bash` rewritten** — now provides flag completion for the
  experiment scripts (`--nh`, `--na`, `--hmin`, `--hmax`, `--axis`,
  `--workers`, `--gpu`, `--live`) and axis-value completion after `--axis`.
  A dispatcher routes completion to the correct handler based on script name.
- `experiments/drop_coin.py`, `drop_cube.py`, `drop_rod.py` — `--live` and
  `--gpu` flags are no longer mutually exclusive.
- **Documentation overhaul**:
  - `docs/architecture.md` — new "Live dashboard" section with architecture
    diagram, "Three implementations" comparison table.
  - `docs/gpu.md` — new Section 8: "`@njit` — the CPU counterpart",
    covering write-once-compile-twice, caching, and performance comparison.
  - `docs/integration.md` — new section on the half-kick artefact and
    correct settle-detection placement within operator splitting.
  - `docs/numerical_methods.md` — new Section 6: "Damping strategies and
    settle detection", covering energy thresholds, damping mechanisms,
    and the force-settle timeout.
  - `docs/parallelism.md` — new section: "Real-time animation: why the main
    thread wins", comparing threaded vs single-thread approaches with
    performance data.
  - `docs/chaos.md` — new Section 9: "Watching chaos unfold: the live
    dashboard", connecting the real-time fill pattern to prediction horizons
    and basin geometry.
  - `docs/physics.md` — new section: "Contact physics and dissipation",
    covering impulse-momentum equations, inertia tensor rotation, Coulomb
    friction, rolling resistance, and shape-dependent resting states.

---

## [0.4.0] — 2026-02-25

### Added
- **GPU-accelerated drop sweeps** via Numba CUDA (`lab/experiments/drop_gpu.py`).
  The entire height × tilt-angle grid runs in parallel on the GPU — one CUDA
  thread per simulation.  On an RTX 3080, a 40×60 grid completes in ~1.4 s
  versus ~55 s on CPU (40× speedup).
  - Quaternion math, leapfrog integration, floor constraint, and outcome
    classification reimplemented as `@cuda.jit(device=True)` device functions.
  - `drop_kernel` — main CUDA kernel, one thread per simulation.
  - `sweep_drop_gpu()` — host function handling memory transfer, grid/block
    sizing, kernel launch, and copy-back.
  - Auto-detection of pip-installed CUDA toolkit via `_setup_cuda_env()`.
- **`--gpu` flag** on `experiments/drop_coin.py`, `drop_cube.py`, `drop_rod.py`
  to run sweeps on the GPU instead of CPU multiprocessing.
- **`--live` flag** for real-time animation during CPU sweeps.  Simulations run
  in a background thread; a `FuncAnimation` loop on the main thread updates the
  outcome map and a histogram panel as results arrive via a `Queue`.
- **GPU tests** (`tests/test_drop_gpu.py`) — 9 tests covering:
  - Device function correctness (quaternion operations vs CPU module).
  - GPU vs CPU result parity (>=40% match on chaotic boundary conditions).
  - Fallback behaviour when CUDA is unavailable.
- **New documentation**:
  - `docs/numerical_methods.md` — IEEE 754 floating point, finite differences,
    convergence order, CFL stability, symplectic structure preservation.
  - `docs/parallelism.md` — processes vs threads, Amdahl's law, Flynn's
    taxonomy (SIMD/SIMT), memory hierarchy, CPU vs GPU comparison.
  - `docs/gpu.md` — CUDA programming model, device functions, thread indexing,
    memory model, divergence, grid sizing, performance analysis, setup guide.
  - `docs/chaos.md` — deterministic chaos, Lyapunov exponents, fractal basins,
    double pendulum, ergodic hypothesis, GPU-vs-CPU divergence as chaos demo.

### Changed
- `docs/physics.md` expanded with Lagrangian mechanics, Legendre transform,
  Noether's theorem, Poisson brackets, and phase space topology.
- `docs/integration.md` expanded with local/global error, modified Hamiltonian
  theory, stability analysis, and adaptive step control.
- `docs/architecture.md` expanded with software design principles (framework vs
  library, plugin architecture, separation of concerns) and GPU execution path.
- `requirements.txt` now includes `numba`.
- Runner scripts (`drop_coin.py`, `drop_cube.py`, `drop_rod.py`) rewritten with
  `--gpu`/`--live` mutual exclusion group and cleaner code organisation.

---

## [0.3.0] — 2026-02-23

### Added
- **Rigid-body drop experiments** — sweep height × tilt-angle for coin, cube,
  and rod, classify the landing outcome (heads/tails/edge, cube face, rod end),
  and plot a colour-coded outcome map.
- `lab/experiments/drop_experiment.py` — core module with:
  - `drop_body()` — single rigid-body drop simulation.
  - `classify()` — dispatcher for coin, cube, and rod outcome classification.
  - `sweep_drop()` — 2D parameter sweep with **CPU multiprocessing**
    (`ProcessPoolExecutor`), configurable worker count, and per-result callback.
  - `plot_drop_map()` — colour-coded image plot with per-shape palettes and
    legend (cube uses paired colours for opposite faces).
- Runner scripts `experiments/drop_coin.py`, `experiments/drop_cube.py`,
  `experiments/drop_rod.py` with CLI arguments (`--nh`, `--na`, `--hmin`,
  `--hmax`, `--axis`, `--workers`).
- **Console progress** — all runner scripts print a header (grid size, height
  range, axis, worker count) and periodic progress lines with elapsed time and
  ETA.

### Changed
- Architecture docs updated with the new experiments layout and usage examples.

---

## [0.2.0] — 2026-02-25

### Fixed
- **Rigid bodies now settle on the floor.** Previously, coins, cubes, and rods
  would spin and rock above the ground indefinitely because the floor constraint
  only applied a normal (bounce) impulse with no mechanism to dissipate
  rotational energy.

### Added
- **Coulomb friction** on `FloorConstraint` — tangential impulse at the contact
  point opposes sliding, capped at `μ × |j_normal|` (default `μ = 0.6`).
- **Rolling resistance** on `FloorConstraint` — each contact event damps angular
  momentum by a small fraction (default 5%), modeling energy lost to
  contact-zone deformation during rocking.
- **Velocity-dependent restitution** — effective restitution drops to zero for
  low-speed contacts (`|v_n| < 0.1 m/s`), preventing infinite micro-bouncing.
- `friction` and `rolling_resistance` parameters on `FloorConstraint`,
  `drop_cube`, `drop_coin`, `drop_rod`, `toss_coin`, `earth_surface`, and
  `capacitor` environment presets.

### Changed
- **Rigid drop animation** rewritten as a dual-panel layout:
  - Left: side-view scene with filled polygon body, textured floor with
    hatching, red CM dot, and a `t = X.XX s` time label.
  - Right: live height-vs-time trace that draws progressively.
  - All shapes (cube, coin, rod) now render as filled 2D cross-sections
    projected from the 3D quaternion orientation.
- `main.py` drop demo uses tuned parameters (`restitution=0.6`, `friction=0.5`,
  `height=2.0`, `duration=4.0`) so the object visibly falls, bounces, and
  settles within the animation window.
- Coin toss deterministic tests (`test_flat_heads`, `test_flat_tails`) now
  explicitly set `friction=0, rolling_resistance=0` to isolate the pure
  restitution physics from contact friction effects.

---

## [0.1.0] — 2026-02-25

Initial release — Hamiltonian physics lab.

### Core framework (`lab/core/`)
- `Hamiltonian` — universal interface: define `H(q, p)` and get equations of
  motion for free.
- `State` — immutable `(q, p)` snapshot.
- `Experiment` — set up initial conditions, pick an integrator, run, and collect
  a `DataSet`.
- `DataSet` — stores time-series `(t, q, p, energy)` with convenience methods
  (`max_energy_error`, etc.).
- **Integrators**: symplectic Leapfrog (Störmer-Verlet), RK4, and adaptive RK45.
- **Quaternion** library for 3D rotations (`exp_map`, `from_axis_angle`,
  `rotate_vector`, `to_rotation_matrix`, `slerp`).

### Pre-built systems (`lab/systems/`)
- **Oscillators**: harmonic, coupled, Duffing, Morse, anharmonic.
- **Pendulums**: simple, double, spherical.
- **Central force**: Kepler, general central potential.
- **Charged particles**: uniform E, uniform B, crossed E×B.
- **Rigid body** subpackage: `RigidBody` (cube, coin, rod factories), `World`
  (leapfrog integrator with quaternion drift), `FloorConstraint`,
  `GravityField`, `UniformElectricField`, `CoulombField`, `DragField`,
  environment presets (`earth_surface`, `vacuum`, `capacitor`),
  `RigidBodyExperiment`, and convenience `drop_*` functions.
- **EM waves**: 1D/2D FDTD Maxwell solver with PML absorbing boundaries,
  material regions, and Gaussian/CW sources.
- **Ray optics**: Hamiltonian geometric optics with graded-index media and
  spherical lens builder.

### Analysis (`lab/analysis/`)
- Energy conservation checks.
- Phase-space portraits.
- Poincaré sections.
- Spectral (FFT) analysis.

### Visualization (`lab/visualization/`)
- **2D animations** (`animate2d.py`): pendulum, double pendulum, orbit, spring,
  coupled spring, charged particle trace, rigid body drop, and generic
  q-vs-time fallback.
- **Field snapshots** (`field_snapshot.py`): FDTD field animation and ray-path
  plots.
- **Interactive** parameter exploration (`interactive.py`).
- **Static plots** (`plots.py`): time-series, phase-space, energy traces.

### Experiments
- **Coin toss** (`lab/experiments/coin_toss.py`): single toss, parameter sweep
  (height × tilt angle), outcome classification (heads/tails/edge), outcome map
  plotting.
- Jupyter notebook (`experiments/coin_toss.ipynb`).

### CLI
- `main.py` — demo launcher with welcome guide (run without arguments).
- Demos: `oscillator`, `coupled`, `pendulum`, `double`, `kepler`, `cyclotron`,
  `drop [cube|coin|rod]`, `emwave`, `rays`.
- Bash tab-completion (`completions.bash`).

### Tests
- 106 pytest unit and integration tests covering quaternions, state, Hamiltonian
  interface, integrators, experiment runner, all system types, rigid body
  dynamics, FDTD, ray optics, and coin toss.

### Documentation (`docs/`)
- `architecture.md` — package layout and data flow.
- `physics.md` — Hamiltonian mechanics as a universal interface.
- `integration.md` — integrator schemes and stability.
- `rotations.md` — quaternion rotations for rigid bodies.
- `systems_catalog.md` — catalog of pre-built Hamiltonians.
