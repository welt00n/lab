# Systems Catalog

A reference for every pre-built system in lab. Each entry shows the Hamiltonian, how to create it, what to explore, and which course it maps to.

---

## Oscillators

### Harmonic oscillator

$$H = \frac{p^2}{2m} + \frac{1}{2}kq^2$$

```python
from lab.systems.oscillators import harmonic
H = harmonic(m=1.0, k=4.0)
```

**Phase portrait**: perfect ellipses. **Explore**: vary $k/m$ to change frequency $\omega = \sqrt{k/m}$.

**Course**: Classical Mechanics

---

### Damped oscillator

$$H = \frac{p^2}{2m} + \frac{1}{2}kq^2 \quad \text{(with friction } \gamma \text{)}$$

```python
from lab.systems.oscillators import damped
H = damped(m=1.0, k=4.0, gamma=0.3)
```

**Phase portrait**: spiralling inward. **Explore**: underdamped ($\gamma < 2\sqrt{km}$) vs overdamped vs critically damped.

---

### Driven oscillator

```python
from lab.systems.oscillators import driven
H = driven(m=1.0, k=4.0, gamma=0.1, F0=1.0, omega_d=2.0)
```

**Explore**: resonance when $\omega_d \approx \omega_0$, transient vs steady state, amplitude vs driving frequency.

---

### Coupled oscillators

$$H = \frac{p_1^2}{2m_1} + \frac{p_2^2}{2m_2} + \frac{1}{2}k_1 q_1^2 + \frac{1}{2}k_2 q_2^2 + \frac{1}{2}k_c(q_1 - q_2)^2$$

```python
from lab.systems.oscillators import coupled
H = coupled(m1=1, m2=1, k1=4, k2=4, kc=1)
```

**Explore**: normal modes, energy transfer between oscillators, beat phenomena.

---

### Duffing oscillator

$$H = \frac{p^2}{2m} + \frac{1}{2}\alpha q^2 + \frac{1}{4}\beta q^4$$

```python
from lab.systems.oscillators import duffing
H = duffing(alpha=-1.0, beta=1.0)
```

**Phase portrait**: double-well potential with separatrix. **Explore**: chaos under driving, bistability.

---

## Pendulums

### Simple pendulum

$$H = \frac{p_\theta^2}{2ml^2} - mgl\cos\theta$$

```python
from lab.systems.pendulums import simple_pendulum
H = simple_pendulum(m=1.0, l=1.0, g=9.81)
```

**Phase portrait**: oscillation inside the separatrix, rotation outside. **Explore**: transition from small-angle (linear) to large-angle (nonlinear), period vs amplitude.

---

### Double pendulum

$$H = T(\theta_1, \theta_2, p_1, p_2) - (m_1+m_2)gl_1\cos\theta_1 - m_2 g l_2 \cos\theta_2$$

```python
from lab.systems.pendulums import double_pendulum
H = double_pendulum(m1=1, m2=1, l1=1, l2=1, g=9.81)
```

**The** canonical example of deterministic chaos. **Explore**: sensitivity to initial conditions, energy surface topology, Poincaré sections.

**Course**: Classical Mechanics

---

### Spherical pendulum

$$H = \frac{p_\theta^2}{2ml^2} + \frac{p_\phi^2}{2ml^2\sin^2\theta} - mgl\cos\theta$$

```python
from lab.systems.pendulums import spherical_pendulum
H = spherical_pendulum(m=1, l=1, g=9.81)
```

**Explore**: precession, conserved angular momentum $p_\phi$, effective potential.

---

## Central force / Orbits

### Kepler orbit

$$H = \frac{p_r^2}{2m} + \frac{p_\theta^2}{2mr^2} - \frac{GMm}{r}$$

```python
from lab.systems.central_force import kepler
H = kepler(m=1, M=1000, G=1)
```

**Explore**: elliptical/circular/parabolic/hyperbolic orbits depending on energy, angular momentum conservation, effective potential $V_{\text{eff}}(r) = p_\theta^2/(2mr^2) - GMm/r$.

**Course**: Classical Mechanics

---

### Precessing orbit

$$V(r) = -\frac{GMm}{r} - \frac{\epsilon}{r^3}$$

```python
from lab.systems.central_force import precessing_orbit
H = precessing_orbit(m=1, M=1000, G=1, eps=0.01)
```

**Explore**: apsidal precession (perihelion advance), compare to pure Kepler.

---

## Charged particles

### Uniform electric field

```python
from lab.systems.charged import uniform_E
H = uniform_E(m=1, charge=1, E=[1, 0, 0])
```

Parabolic trajectories (constant acceleration). **Course**: Electromagnetism

---

### Uniform magnetic field (cyclotron)

```python
from lab.systems.charged import uniform_B
H = uniform_B(m=1, charge=1, B=[0, 0, 1])
```

**Explore**: circular motion, Larmor radius $r_L = mv_\perp / |q|B$, cyclotron frequency $\omega_c = |q|B/m$.

**Course**: Electromagnetism

---

### Crossed E×B fields

```python
from lab.systems.charged import crossed_EB
H = crossed_EB(m=1, charge=1, E=[1, 0, 0], B=[0, 0, 1])
```

**Explore**: E×B drift velocity $v_d = E \times B / B^2$, cycloid trajectories.

**Course**: Electromagnetism

---

## Electromagnetic waves (FDTD)

### 1D wave propagation

```python
from lab.systems.emwave import FDTDGrid1D, gaussian_pulse

grid = FDTDGrid1D(nx=400, dx=0.01)
grid.add_source(gaussian_pulse(position_index=50))
grid.enable_pml(width=20)
data = grid.run(nsteps=800)
```

**Explore**: propagation, reflection from dielectric interfaces, PML absorption, standing waves.

---

### 2D wave propagation

```python
from lab.systems.emwave import FDTDGrid2D, point_source_2d

grid = FDTDGrid2D(nx=200, ny=200, dx=0.01)
grid.add_source(point_source_2d(100, 100, frequency=10))
data = grid.run(nsteps=500)
```

**Explore**: diffraction, interference, waveguides, dielectric slabs.

**Course**: Electromagnetism & Optics

---

## Ray optics

### Lens (geometric optics)

```python
from lab.systems.ray_optics import ray_hamiltonian, spherical_lens, launch_fan
from lab.core.experiment import Experiment

n = spherical_lens(center=[3, 0], radius=0.8, n_lens=1.5)
H = ray_hamiltonian(n, ndim=2)

rays = launch_fan(n, origin=[0, 0], angles=np.linspace(-0.3, 0.3, 20))
datasets = []
for q0, p0 in rays:
    exp = Experiment(H, q0=q0, p0=p0, dt=0.005, duration=6.0)
    datasets.append(exp.run())
```

**Explore**: focal length, aberrations, graded-index fibers, Snell's law.

**Course**: Optics

---

## Rigid bodies

### Dropping objects

```python
from lab.systems.rigid_body import drop_cube

exp = drop_cube(side=0.3, height=2.0, restitution=0.8, dt=0.001, duration=5.0)
data = exp.run()
```

Available: `drop_cube`, `drop_coin`, `drop_rod`. **Explore**: which face a cube lands on vs initial tilt, coin toss probability, angular momentum transfer on bounce.

**Course**: Classical Mechanics

---

## Adding your own system

Any system with a Hamiltonian $H(q, p)$:

```python
from lab.core.hamiltonian import Hamiltonian

H = Hamiltonian(
    ndof=1,
    kinetic=lambda q, p: p[0]**2 / 2,
    potential=lambda q, p: q[0]**4 - 2*q[0]**2,
    name="double well",
    coords=["x"],
)
```

Then use `Experiment`, `DataSet`, and all analysis/visualization tools as normal.
