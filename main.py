"""
lab — a Hamiltonian physics lab.
"""

import sys
import numpy as np


def demo_oscillator():
    from lab.systems.oscillators import harmonic
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = harmonic(m=1.0, k=4.0)
    data = Experiment(H, q0=[2.0], p0=[0.0], dt=0.01, duration=20.0).run()
    print(f"Harmonic oscillator — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_coupled():
    from lab.systems.oscillators import coupled
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = coupled(m1=1, m2=1, k1=2, k2=2, kc=0.5)
    data = Experiment(H, q0=[1.0, 0.0], p0=[0.0, 0.0],
                      dt=0.01, duration=30.0).run()
    print(f"Coupled oscillators — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_pendulum():
    from lab.systems.pendulums import simple_pendulum
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = simple_pendulum(m=1, l=1, g=9.81)
    data = Experiment(H, q0=[2.5], p0=[0.0], dt=0.005, duration=20.0).run()
    print(f"Simple pendulum — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_double_pendulum():
    from lab.systems.pendulums import double_pendulum
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = double_pendulum(m1=1, m2=1, l1=1, l2=1, g=9.81)
    data = Experiment(H, q0=[2.0, 2.0], p0=[0.0, 0.0],
                      dt=0.002, duration=20.0, integrator="rk4").run()
    print(f"Double pendulum — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_kepler():
    from lab.systems.central_force import kepler
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = kepler(m=1, M=100, G=1)
    data = Experiment(H, q0=[5.0, 0.0], p0=[0.0, 18.0],
                      dt=0.002, duration=80.0).run()
    print(f"Kepler orbit — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_cyclotron():
    from lab.systems.charged import uniform_B
    from lab.core.experiment import Experiment
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    H = uniform_B(m=1, charge=1, B=[0, 0, 1])
    data = Experiment(H, q0=[0, 0, 0], p0=[1, 0, 0],
                      dt=0.005, duration=30.0, integrator="rk4").run()
    print(f"Cyclotron — energy error: {data.max_energy_error():.2e}")
    anim = animate(data)
    plt.show()


def demo_drop(body_type="cube"):
    from lab.systems.rigid_body import drop_cube, drop_coin, drop_rod
    from lab.core import quaternion as quat
    from lab.visualization.animate2d import animate
    import matplotlib.pyplot as plt

    builders = {"cube": drop_cube, "coin": drop_coin, "rod": drop_rod}
    builder = builders.get(body_type, drop_cube)
    orientation = quat.from_axis_angle([1, 1, 0], 0.4)

    exp = builder(height=2.0, restitution=0.6, friction=0.5, dt=0.001,
                  duration=4.0, orientation=orientation)
    data = exp.run(progress=True)
    print(f"{body_type} drop — final height: {data.q[-1, 1]:.3f} m")
    anim = animate(data)
    plt.show()


def demo_emwave():
    from lab.systems.emwave import FDTDGrid1D, gaussian_pulse
    from lab.visualization.field_snapshot import animate_1d
    import matplotlib.pyplot as plt

    grid = FDTDGrid1D(nx=400, dx=0.01)
    grid.add_source(gaussian_pulse(50, amplitude=1.0, width=20, delay=40))
    grid.set_material(200, 300, epsilon=4.0)
    grid.enable_pml(width=30)
    print("Running 1D FDTD — Gaussian pulse hitting a dielectric slab...")
    data = grid.run(nsteps=600, snapshot_interval=3)
    print(f"FDTD — {data.nsteps} snapshots captured")
    anim = animate_1d(data)
    plt.show()


def demo_rays():
    from lab.systems.ray_optics import (
        ray_hamiltonian, spherical_lens, launch_fan,
    )
    from lab.core.experiment import Experiment
    from lab.visualization.field_snapshot import plot_ray_paths
    import matplotlib.pyplot as plt

    n = spherical_lens(center=[3.0, 0.0], radius=1.0, n_lens=1.8, n_outside=1.0)
    H = ray_hamiltonian(n, ndim=2)

    angles = np.linspace(-0.35, 0.35, 25)
    ics = launch_fan(n, origin=[0.0, 0.0], angles=angles)

    datasets = []
    for q0, p0 in ics:
        data = Experiment(H, q0=q0, p0=p0, dt=0.01, duration=6.0,
                          integrator="rk4").run()
        datasets.append(data)

    print(f"Ray optics — {len(datasets)} rays through a spherical lens")
    plot_ray_paths(datasets, n_func=n, xlim=(-0.5, 7), ylim=(-3, 3))
    plt.title("Rays through a spherical lens")
    plt.tight_layout()
    plt.show()


DEMOS = {
    "oscillator": demo_oscillator,
    "coupled": demo_coupled,
    "pendulum": demo_pendulum,
    "double": demo_double_pendulum,
    "kepler": demo_kepler,
    "cyclotron": demo_cyclotron,
    "drop": demo_drop,
    "emwave": demo_emwave,
    "rays": demo_rays,
}


def welcome():
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ⚛  lab — Hamiltonian Physics Lab                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{RESET}

  A framework for simulating classical systems through the
  universal language of Hamiltonian mechanics: {BOLD}H(q, p){RESET}.

{BOLD}{YELLOW}  Demos{RESET}  {DIM}(each opens an animated matplotlib window){RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  {GREEN}python main.py oscillator{RESET}    Spring-mass animation + trace
  {GREEN}python main.py coupled{RESET}       Two coupled oscillators
  {GREEN}python main.py pendulum{RESET}      Swinging pendulum with trail
  {GREEN}python main.py double{RESET}        Double pendulum — chaos
  {GREEN}python main.py kepler{RESET}        Kepler orbit tracing
  {GREEN}python main.py cyclotron{RESET}     Charged particle in B field
  {GREEN}python main.py drop{RESET}          Rigid body drop  {DIM}[cube|coin|rod]{RESET}
  {GREEN}python main.py emwave{RESET}        EM pulse hitting a dielectric slab
  {GREEN}python main.py rays{RESET}          Light rays through a spherical lens

{BOLD}{YELLOW}  Using the library{RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  {MAGENTA}from lab.core.hamiltonian import Hamiltonian
  from lab.core.experiment  import Experiment{RESET}

  Define H(q, p), pick initial conditions and an integrator,
  and let the Experiment runner produce a DataSet you can
  plot, animate, or analyse.

{BOLD}{YELLOW}  Available systems{RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  lab.systems.{GREEN}oscillators{RESET}     harmonic, coupled, duffing, ...
  lab.systems.{GREEN}pendulums{RESET}       simple, double, spherical
  lab.systems.{GREEN}central_force{RESET}   kepler, general central
  lab.systems.{GREEN}charged{RESET}         uniform E, uniform B, crossed E×B
  lab.systems.{GREEN}rigid_body{RESET}      3D rigid bodies with contact
  lab.systems.{GREEN}emwave{RESET}          FDTD Maxwell solver (1D, 2D)
  lab.systems.{GREEN}ray_optics{RESET}      Hamiltonian geometric optics

{BOLD}{YELLOW}  Experiments{RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  See {CYAN}experiments/{RESET} for Jupyter notebooks (e.g. coin toss).

{BOLD}{YELLOW}  Tests{RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  {GREEN}python -m pytest tests/ -v{RESET}

{BOLD}{YELLOW}  Docs{RESET}
{DIM}  ─────────────────────────────────────────────────────────{RESET}
  See {CYAN}docs/{RESET} for architecture, physics, and integration guides.
""")


def main():
    args = sys.argv[1:]

    if not args:
        welcome()
        return

    name = args[0]
    if name == "drop":
        body_type = args[1] if len(args) > 1 else "cube"
        demo_drop(body_type)
    elif name in DEMOS:
        DEMOS[name]()
    else:
        print(f"Unknown demo: {name!r}\n")
        print(f"Available demos: {', '.join(DEMOS.keys())}")
        print("Run without arguments for full help.")


if __name__ == "__main__":
    main()
