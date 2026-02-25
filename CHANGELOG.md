# Changelog

All notable changes to **lab** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

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
