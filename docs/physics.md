# The Hamiltonian as Universal Interface

## Why Hamiltonian mechanics?

Every system in classical physics — a ball on a spring, a planet orbiting a star, a charged particle spiralling in a magnetic field, a light ray bending through glass — obeys the same mathematical structure. The Hamiltonian $H(q, p)$ is a single function that encodes *everything* about a system's dynamics.

Given $H$, you get the equations of motion for free:

$$
\dot{q}_i = \frac{\partial H}{\partial p_i} \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

That's it. Two equations. They apply to *every* classical system, regardless of how many particles, dimensions, or forces are involved.

## What are $q$ and $p$?

**Generalized coordinates** $q$ describe the configuration of the system. They can be positions, angles, field amplitudes — anything that specifies "where" the system is.

**Conjugate momenta** $p$ describe how the system is moving. For a simple particle, $p = mv$ (mass times velocity). For a pendulum, $p_\theta = ml^2 \dot{\theta}$ (angular momentum). The relationship between $p$ and velocity comes from $H$ itself.

Together, $(q, p)$ define a point in **phase space** — the complete state of the system at one instant.

## The Hamiltonian for common systems

### Free particle

$$
H = \frac{p^2}{2m}
$$

No potential, so $\dot{p} = 0$ (momentum is conserved) and $\dot{q} = p/m$ (constant velocity).

### Harmonic oscillator

$$
H = \frac{p^2}{2m} + \frac{1}{2}k q^2
$$

Kinetic + potential energy. Phase-space trajectories are ellipses — the system oscillates forever with constant energy.

### Pendulum

$$
H = \frac{p_\theta^2}{2ml^2} - mgl\cos\theta
$$

Unlike the linearized version ($\sin\theta \approx \theta$), this captures the full nonlinear dynamics: large-amplitude oscillations, the separatrix between oscillation and rotation, and the unstable equilibrium at the top.

### Kepler orbit

$$
H = \frac{p_r^2}{2m} + \frac{p_\theta^2}{2mr^2} - \frac{GMm}{r}
$$

In polar coordinates $(r, \theta)$. The $p_\theta^2 / r^2$ term is the centrifugal barrier. Bounded orbits are ellipses; unbound orbits are hyperbolas.

### Charged particle in electromagnetic fields

$$
H = \frac{|\vec{p} - q\vec{A}(\vec{r})|^2}{2m} + q\phi(\vec{r})
$$

where $\vec{A}$ is the vector potential and $\phi$ is the scalar potential. The canonical momentum $\vec{p}$ is *not* $m\vec{v}$ — it includes the field contribution $q\vec{A}$. This is how the Hamiltonian handles velocity-dependent forces (like the Lorentz force from a magnetic field).

### Rigid body

$$
H = \frac{|\vec{p}|^2}{2m} + \frac{1}{2}\vec{L}^T I^{-1} \vec{L} + V(\vec{r})
$$

Translational kinetic energy + rotational kinetic energy (where $\vec{L}$ is angular momentum and $I$ is the inertia tensor) + potential energy. Orientation is tracked with quaternions to avoid gimbal lock.

### Geometric optics (ray tracing)

$$
H = \frac{|\vec{p}|}{n(\vec{q})}
$$

Light rays in a medium with refractive index $n(\vec{q})$. Hamilton's equations reproduce Snell's law at interfaces and curved paths in graded-index media. This is the connection between wave optics and particle mechanics — light behaves like a Hamiltonian particle.

## Energy conservation

For any system where $H$ doesn't depend explicitly on time:

$$
\frac{dH}{dt} = \frac{\partial H}{\partial q}\dot{q} + \frac{\partial H}{\partial p}\dot{p} = \frac{\partial H}{\partial q}\frac{\partial H}{\partial p} - \frac{\partial H}{\partial p}\frac{\partial H}{\partial q} = 0
$$

Energy is *exactly* conserved. This isn't an approximation — it's a mathematical identity. Numerically, we use a **symplectic integrator** (leapfrog) that preserves this structure, so energy stays bounded even over millions of time steps rather than drifting.

## Phase space and Liouville's theorem

The state $(q, p)$ evolves as a flow in phase space. Liouville's theorem says this flow is **incompressible**: volumes in phase space are preserved. This is why symplectic integrators are the right tool — they respect this geometric structure by construction.

For 1-DOF systems, you can visualize the entire dynamics as contour lines of $H$ in the $(q, p)$ plane. Each contour is a possible trajectory. The topology of these contours reveals everything: stable equilibria (elliptic fixed points), unstable equilibria (hyperbolic fixed points), separatrices (boundaries between qualitatively different motions), and chaos (when contours dissolve into a tangle for systems with 2+ DOF).

## Non-conservative systems

Damping, driving forces, and friction break energy conservation. The framework handles these by modifying the gradients $\partial H/\partial q$ to include non-conservative terms. The integrator still works — it just won't conserve energy (which is physically correct). You'll see the energy plot drift or oscillate, telling you about energy dissipation or injection.

## Electromagnetic waves (FDTD)

Maxwell's equations can be written in Hamiltonian form where the **fields** are the dynamical variables:

$$
H = \frac{1}{2}\int \left(\epsilon|\vec{E}|^2 + \frac{|\vec{B}|^2}{\mu}\right) dV
$$

The FDTD algorithm discretizes this on a Yee lattice, updating $\vec{E}$ and $\vec{B}$ in a leapfrog pattern — the same symplectic idea as the particle integrator, applied to fields instead of particles.

## How the framework uses all this

1. **You define** $H(q, p)$ (or pick a pre-built system)
2. **The integrator** computes $\partial H/\partial q$ and $\partial H/\partial p$ (analytically or numerically) and advances the state
3. **The experiment runner** records the trajectory $(t, q, p, H)$
4. **Analysis tools** compute energy conservation, phase portraits, Poincaré sections, and power spectra — all from the same DataSet
5. **Visualization** animates the results, with system-specific renderers (pendulums swing, orbits trace ellipses, waves propagate)

The Hamiltonian is the universal interface between *your physics knowledge* and *the computer's ability to simulate*.
