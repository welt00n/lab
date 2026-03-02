# The Hamiltonian as Universal Interface

## Why Hamiltonian mechanics?

Every system in classical physics — a ball on a spring, a planet orbiting a star, a charged particle spiralling in a magnetic field, a light ray bending through glass — obeys the same mathematical structure. The Hamiltonian $H(q, p)$ is a single function that encodes *everything* about a system's dynamics.

Given $H$, you get the equations of motion for free:

$$
\dot{q}_i = \frac{\partial H}{\partial p_i} \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

That's it. Two equations. They apply to *every* classical system, regardless of how many particles, dimensions, or forces are involved.
$$
\frac{dH}{dt} = \frac{\partial H}{\partial q}\dot{q} + \frac{\partial H}{\partial p}\dot{p} = \frac{\partial H}{\partial q}\frac{\partial H}{\partial p} - \frac{\partial H}{\partial p}\frac{\partial H}{\partial q} = 0
$$

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

Here, "drift" means a spurious, long‑term change in the computed value of the Hamiltonian that does not occur in the true conservative dynamics. Non‑symplectic integrators (for example explicit Euler or many Runge–Kutta schemes) often produce secular drift: energy slowly grows or decays over many steps and trajectories spiral away from the true $H$‑contours (artificial heating or cooling). Symplectic integrators instead preserve the symplectic form and conserve a nearby "modified Hamiltonian": numerical energy typically oscillates with a small, bounded amplitude (controlled by the timestep) rather than showing unbounded secular drift. Caveats: genuine physical dissipation or explicit time dependence of $H$ causes true energy change; floating‑point roundoff can still accumulate over extremely long runs; reducing the timestep lowers the oscillation amplitude.


## Phase space and Liouville's theorem

The state $(q, p)$ evolves as a flow in phase space. Liouville's theorem says this flow is **incompressible**: volumes in phase space are preserved. This is why symplectic integrators are the right tool — they respect this geometric structure by construction.

The variables $(q, p)$ live in **phase space**, which has a built-in geometric structure called the **symplectic form**. This structure is what Liouville's theorem expresses, and it's what symplectic integrators preserve.

A slightly deeper look: the symplectic form is the closed, nondegenerate 2-form
$$
\omega = \sum_{i=1}^n dq_i \wedge dp_i,
$$
which defines an oriented area element on each conjugate $(q_i,p_i)$ plane. The Hamiltonian vector field $X_H$ satisfies
$$
\iota_{X_H}\omega = dH
$$
and hence preserves the form: $\mathcal{L}_{X_H}\omega=0$, equivalently the flow $\phi_t$ obeys $\phi_t^*\omega=\omega$. This is the geometric content of Liouville's theorem: phase‑space volumes are preserved (the flow is divergence‑free and has Jacobian determinant one).

When a numerical integrator is symplectic its discrete time‑step map $\Phi_h$ exactly preserves $\omega$ (i.e. $\Phi_h^*\omega=\omega$). Such maps therefore preserve area/volume and the qualitative phase‑space geometry (invariant tori, fixed points, separatrices), which is why they reproduce long‑time behaviour better than non‑symplectic methods.

Backward‑error analysis gives a precise explanation: a symplectic integrator with step size $h$ can be seen as exactly integrating the flow of a nearby Hamiltonian $\tilde H = H + h^k H_k + \cdots$. As a result the numerical energy oscillates around the true energy with a small, bounded amplitude (controlled by $h$ and the method order) rather than showing secular drift. By contrast, non‑symplectic methods typically fail to preserve $\omega$ and introduce spurious dissipation or growth of invariants over long times.

Practical caveats: finite‑precision round‑off can still accumulate on very long runs, and physical non‑conservative forces (or operator‑split dissipative corrections) change energy for real reasons — symplectic structure only protects the conservative part.
For 1-DOF systems, you can visualize the entire dynamics as contour lines of $H$ in the $(q, p)$ plane. Each contour is a possible trajectory. The topology of these contours reveals everything: stable equilibria (elliptic fixed points), unstable equilibria (hyperbolic fixed points), separatrices (boundaries between qualitatively different motions), and chaos (when contours dissolve into a tangle for systems with 2+ DOF).

## Units convention

All quantities in this framework are in **SI units**: metres, kilograms, seconds, radians. Energies are in joules, momenta in kg·m/s, angular momenta in kg·m²/s. See [realistic_parameters.md](realistic_parameters.md) for the specific object constants (US quarter, 16 mm die) and their derivations.

## Non-conservative systems

Damping, driving forces, and friction break energy conservation. Hamiltonian mechanics handles this cleanly: the equations of motion acquire additional terms that cannot be derived from $H$ alone.

**Drag forces** modify the momentum equation directly. The `DragField` in `lab/systems/rigid_body/fields.py` adds a velocity-dependent force:

$$
\dot{\vec{p}} = -\frac{\partial H}{\partial \vec{q}} - b\,\frac{\vec{p}}{m}
$$

where $b$ is the drag coefficient. The extra term is *not* a gradient of any scalar — it has no potential — so it breaks the symplectic structure. Energy decreases monotonically.

**Contact constraints** are non-smooth: the floor exerts zero force when the body is airborne and an impulsive force at contact. The `FloorConstraint` in `lab/systems/rigid_body/constraints.py` applies impulse-based corrections (restitution, friction, rolling resistance) that dissipate energy in discrete events rather than continuously. See [contact_model.md](contact_model.md) for a full derivation.

The integrator still works in both cases — it applies the conservative (Hamiltonian) part symplectically and the dissipative part as an operator-splitting correction. Energy is no longer conserved, which is physically correct. The energy trace from `lab/analysis/energy.py` becomes a diagnostic: monotonic decrease means pure dissipation; stepwise drops indicate individual contact events.

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

---

## From Newton to Lagrange to Hamilton

The three pillars of classical mechanics — Newtonian, Lagrangian, and Hamiltonian — aren't competing theories. They're the *same* physics written in progressively more powerful mathematical languages. Understanding how each leads to the next reveals why Hamiltonian mechanics sits at the top.

### Newton: forces and constraints

Newton's second law is where everyone starts:

$$
\vec{F} = m\vec{a}
$$

For a single particle in free space, this is fine. But the moment you add constraints — a bead on a wire, a ball rolling on a surface, a double pendulum — you're in trouble. You need to figure out all the constraint forces (normal forces, tensions), decompose everything into components, and solve a system of second-order vector equations. For $N$ particles in 3D with $k$ constraints, you have $3N - k$ true degrees of freedom but $3N$ equations laced together by forces you don't care about.

### Lagrange: action and generalized coordinates

Lagrange's insight was to work directly with the $n = 3N - k$ independent degrees of freedom $q_1, \ldots, q_n$ — the **generalized coordinates**. Define the **Lagrangian**:

$$
L(q, \dot{q}, t) = T - V
$$

where $T$ is kinetic energy and $V$ is potential energy, both expressed in terms of $q$ and $\dot{q}$. Constraint forces disappear entirely because the generalized coordinates *already* satisfy the constraints.

The dynamics follow from the **principle of least action** (more precisely, *stationary* action). The **action** is:

$$
S[q] = \int_{t_1}^{t_2} L(q, \dot{q}, t)\, dt
$$

The physical trajectory is the one for which $S$ is stationary under variations $q(t) \to q(t) + \delta q(t)$ with fixed endpoints. Setting $\delta S = 0$ yields the **Euler–Lagrange equations**:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0 \qquad i = 1, \ldots, n
$$

One equation per degree of freedom, no constraint forces, any coordinate system you like. This is already a huge improvement over Newton.

**Example — simple pendulum.** With generalized coordinate $\theta$:

$$
L = \tfrac{1}{2}ml^2\dot{\theta}^2 + mgl\cos\theta
$$

The Euler–Lagrange equation gives $ml^2\ddot{\theta} = -mgl\sin\theta$, i.e. $\ddot{\theta} = -(g/l)\sin\theta$ — no need to decompose gravity into radial and tangential components.

### The Legendre transform: from $L$ to $H$

Lagrangian mechanics uses $(q, \dot{q})$ as the state variables. The Hamiltonian formulation switches to $(q, p)$ by trading velocity $\dot{q}$ for **conjugate momentum** $p$:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i}
$$

This is the **Legendre transform** — a change of independent variable from $\dot{q}$ to $p$. The Hamiltonian is constructed as:

$$
H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)
$$

where $\dot{q}$ on the right side is eliminated in favour of $p$ using the definition above. For "natural" systems (kinetic energy quadratic in velocities, potential independent of velocities), this reduces to $H = T + V$ — the total energy.

**Example — simple pendulum.** From $L = \tfrac{1}{2}ml^2\dot{\theta}^2 + mgl\cos\theta$:

$$
p_\theta = ml^2 \dot{\theta} \qquad \Longrightarrow \qquad \dot{\theta} = \frac{p_\theta}{ml^2}
$$

$$
H = p_\theta \dot{\theta} - L = \frac{p_\theta^2}{2ml^2} - mgl\cos\theta
$$

which is the expression from the pendulum section above.

### Why Hamiltonian form wins computationally

The Euler–Lagrange equations are $n$ **second-order** ODEs. Hamilton's equations are $2n$ **first-order** ODEs:

$$
\dot{q}_i = \frac{\partial H}{\partial p_i} \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}
$$

First-order systems are what numerical integrators actually solve. You could always rewrite a second-order system as first-order by introducing auxiliary variables — but in the Hamiltonian case, the first-order structure is *natural*. The variables $(q, p)$ live in **phase space**, which has a built-in geometric structure called the **symplectic form**. This structure is what Liouville's theorem (section above) expresses, and it's what symplectic integrators preserve.

The chain from Newton to Hamilton:

| Formulation | State variables | Equations | Order | Constraints |
|---|---|---|---|---|
| Newton | $\vec{r}, \vec{v}$ | $\vec{F} = m\vec{a}$ | 2nd | Must track constraint forces |
| Lagrange | $q, \dot{q}$ | $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} = \frac{\partial L}{\partial q}$ | 2nd | Eliminated by coordinate choice |
| Hamilton | $q, p$ | $\dot{q} = \partial_p H,\; \dot{p} = -\partial_q H$ | 1st | Eliminated; symplectic structure |

Each step strips away inessential detail and exposes more of the underlying geometry.

---

## Noether's theorem

If the Hamiltonian framework is the *language* of classical mechanics, Noether's theorem is its most profound *sentence*. It connects symmetries — things you can do to a system without changing the physics — to conservation laws — quantities that remain constant in time.

### Statement

**Noether's theorem (informal):** Every continuous symmetry of the action implies a conserved quantity, and conversely.

More precisely: if the Lagrangian (or equivalently, the Hamiltonian) is invariant under a continuous one-parameter family of transformations $q \to q + \epsilon\, \delta q$, then the quantity

$$
Q = \sum_i \frac{\partial L}{\partial \dot{q}_i}\, \delta q_i = \sum_i p_i\, \delta q_i
$$

is conserved along trajectories: $\dot{Q} = 0$.

### The three classical conservation laws

Each familiar conservation law corresponds to a specific symmetry:

**Time translation symmetry** $\longleftrightarrow$ **Energy conservation.**
If $H$ doesn't depend explicitly on time ($\partial H / \partial t = 0$), then $H$ itself is the conserved quantity. We proved this in the energy conservation section above: $dH/dt = 0$.

**Space translation symmetry** $\longleftrightarrow$ **Momentum conservation.**
If $H$ doesn't depend on a particular coordinate $q_i$ (such a coordinate is called **cyclic** or **ignorable**), then:

$$
\dot{p}_i = -\frac{\partial H}{\partial q_i} = 0
$$

The conjugate momentum $p_i$ is conserved. For example, if a system is invariant under shifts in $x$, then the $x$-momentum $p_x$ is constant.

**Rotational symmetry** $\longleftrightarrow$ **Angular momentum conservation.**
If $H$ is unchanged under rotations about some axis — meaning it doesn't depend on the azimuthal angle $\phi$ — then:

$$
\dot{p}_\phi = -\frac{\partial H}{\partial \phi} = 0
$$

The angular momentum $p_\phi$ (often written $L_z$ or $J_z$) is conserved. This is why Kepler orbits stay in a plane.

### Why this matters

Noether's theorem is arguably the deepest result in all of classical mechanics. It tells you that conservation laws aren't accidents or approximations — they are *consequences of symmetry*. And symmetry can be *read off* from the Hamiltonian. If $q_i$ doesn't appear in $H$, you get a conserved $p_i$ for free. No calculation needed beyond inspection.

This has a practical payoff for simulation: every conserved quantity is an **independent check** on the integrator. If your code says energy is drifting when $H$ has no explicit time dependence, there's a bug — not in the physics, but in the numerics.

### Example: the harmonic oscillator

The oscillator Hamiltonian:

$$
H = \frac{p^2}{2m} + \frac{1}{2}kq^2
$$

does not depend on time explicitly ($\partial H / \partial t = 0$), so by Noether's theorem, energy $E = H$ is conserved. The phase-space trajectories are closed ellipses — a particle that starts on one ellipse stays on it forever. If your simulation's energy plot shows drift, the integrator is breaking the symmetry that guarantees conservation.

Note that $H$ *does* depend on $q$, so linear momentum $p$ is **not** conserved — the spring exerts a restoring force $\dot{p} = -kq$.

---

## Poisson brackets

Poisson brackets are the algebraic backbone of Hamiltonian mechanics. They encode the same information as Hamilton's equations but in a coordinate-free, algebraic language that generalizes beautifully — all the way to quantum mechanics.

### Definition

For any two functions $f(q, p)$ and $g(q, p)$ on phase space, the **Poisson bracket** is:

$$
\{f, g\} = \sum_{i=1}^{n} \left( \frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i} \right)
$$

This is an antisymmetric bilinear operation: $\{f, g\} = -\{g, f\}$ and $\{f, f\} = 0$.

### Hamilton's equations, rewritten

The time evolution of *any* function $f(q, p)$ along a trajectory is:

$$
\dot{f} = \{f, H\}
$$

To recover Hamilton's equations, just plug in $f = q_i$ and $f = p_i$:

$$
\dot{q}_i = \{q_i, H\} = \frac{\partial H}{\partial p_i} \qquad \dot{p}_i = \{p_i, H\} = -\frac{\partial H}{\partial q_i}
$$

So the Hamiltonian $H$ is the **generator of time evolution** — the Poisson bracket with $H$ is the operation "take the time derivative."

### Conservation laws, rewritten

A quantity $f(q, p)$ is conserved if and only if:

$$
\{f, H\} = 0
$$

This is the Poisson-bracket version of Noether's theorem. If $f$ Poisson-commutes with $H$, it's a constant of the motion. Energy conservation itself is $\{H, H\} = 0$ — trivially true by antisymmetry.

### Fundamental Poisson brackets

The coordinates and momenta satisfy the **canonical commutation relations**:

$$
\{q_i, q_j\} = 0 \qquad \{p_i, p_j\} = 0 \qquad \{q_i, p_j\} = \delta_{ij}
$$

where $\delta_{ij}$ is the Kronecker delta (1 if $i = j$, 0 otherwise). These relations *define* what it means for $(q, p)$ to be canonical coordinates. A transformation $(q, p) \to (Q, P)$ that preserves these brackets is a **canonical transformation** — the Hamiltonian framework's version of a change of variables.

### The bridge to quantum mechanics

Here is one of the most beautiful correspondences in all of physics. Dirac's **canonical quantization** prescription says: replace Poisson brackets with commutators, scaled by $i\hbar$:

$$
\{f, g\}_\text{classical} \;\longrightarrow\; \frac{1}{i\hbar}[\hat{f}, \hat{g}]_\text{quantum}
$$

The canonical Poisson brackets $\{q_i, p_j\} = \delta_{ij}$ become the **Heisenberg commutation relations**:

$$
[\hat{q}_i, \hat{p}_j] = i\hbar\, \delta_{ij}
$$

Hamilton's equation $\dot{f} = \{f, H\}$ becomes the **Heisenberg equation of motion**:

$$
\frac{d\hat{f}}{dt} = \frac{1}{i\hbar}[\hat{f}, \hat{H}]
$$

The entire algebraic structure of classical mechanics survives intact in quantum mechanics — with Poisson brackets promoted to commutators. This is why learning Hamiltonian mechanics pays off enormously when you reach quantum mechanics: the formalism is already in your hands, and quantization is a *translation*, not a revolution.

### Example: angular momentum algebra

For a 3D system, the angular momentum components $L_x$, $L_y$, $L_z$ satisfy:

$$
\{L_x, L_y\} = L_z \qquad \{L_y, L_z\} = L_x \qquad \{L_z, L_x\} = L_y
$$

Under canonical quantization, these become the quantum angular momentum commutators $[\hat{L}_x, \hat{L}_y] = i\hbar \hat{L}_z$ (and cyclic permutations) — the starting point for the theory of spin, atomic orbitals, and selection rules.

---

## Phase space topology

We've been drawing phase-space pictures — contour plots of $H(q, p)$ — throughout this document. Now let's look at what those pictures are actually *telling us* about the qualitative behaviour of a system.

### Fixed points

A **fixed point** (or equilibrium) is a point $(q_0, p_0)$ where $\dot{q} = 0$ and $\dot{p} = 0$ simultaneously, i.e. where both partial derivatives of $H$ vanish. There are two main types:

**Elliptic (stable, center).** Nearby trajectories form closed loops around the fixed point. The system oscillates. Linearizing Hamilton's equations around an elliptic point gives purely imaginary eigenvalues $\pm i\omega$. Example: the bottom of a pendulum ($\theta = 0$, $p_\theta = 0$).

**Hyperbolic (unstable, saddle).** Nearby trajectories approach along one direction and flee along another. The eigenvalues are real: $\pm \lambda$. Example: the top of a pendulum ($\theta = \pi$, $p_\theta = 0$). A small perturbation either sends the pendulum swinging over or back.

### Separatrices

A **separatrix** is a special trajectory that connects hyperbolic fixed points (or a hyperbolic point to itself). It's the boundary between qualitatively different types of motion.

For the pendulum, the separatrix is the curve $H = mgl$ (the energy at the unstable equilibrium). It divides phase space into three regions:

- **Inside the separatrix (libration):** The pendulum swings back and forth without going over the top. Trajectories are closed loops around the elliptic point.
- **Outside the separatrix (rotation):** The pendulum has enough energy to continuously spin in one direction. Trajectories are open curves that wrap around the cylindrical phase space.
- **On the separatrix:** The pendulum asymptotically approaches the inverted position. It takes infinite time to reach the top — the motion "just barely" connects the two unstable equilibria.

The energy value of the separatrix is a critical threshold. Below it, the dynamics are qualitatively one thing (oscillation); above it, another (rotation). This kind of topological classification — asking "what types of motion exist?" rather than "what is the exact trajectory?" — is one of the most powerful aspects of phase-space analysis.

### Invariant tori and integrability

For a system with $n$ degrees of freedom and $n$ independent conserved quantities in involution (their pairwise Poisson brackets all vanish), a theorem due to Liouville and Arnold says the motion is confined to $n$-dimensional **invariant tori** in the $2n$-dimensional phase space.

Think of the simplest case: a 1-DOF oscillator has $n = 1$ conserved quantity (energy), and the "torus" is just a closed loop in the $(q, p)$ plane. For $n = 2$ (say, two uncoupled oscillators), the torus is a 2D surface — trajectories wind around it like thread on a donut.

**Integrable systems** — those with enough conserved quantities to fill phase space with tori — are the "nice" systems where motion is quasi-periodic and predictable.

### KAM theory: what happens when tori break

What if you perturb an integrable system slightly? The **Kolmogorov–Arnold–Moser (KAM) theorem** gives a remarkable answer: *most* tori survive, slightly deformed, as long as the perturbation is small enough and the frequencies on the torus are "sufficiently irrational" (not in resonance).

But tori with rational frequency ratios — where the frequencies are commensurate — *do* break. In the gaps left behind, trajectories wander erratically. This is the onset of **chaos**: not a sudden explosion, but a gradual dissolution of orderly tori into tangled, space-filling trajectories.

For small perturbations, the surviving KAM tori act as barriers that confine chaotic trajectories to narrow stochastic layers around the destroyed resonant tori. As the perturbation grows, more tori break, the chaotic layers widen and merge, and eventually the motion becomes globally chaotic.

### Connection to the simulation framework

Phase-space topology is exactly what our analysis tools compute:

- **Phase portraits** show the contour lines of $H$ — the tori, separatrices, and fixed points.
- **Poincaré sections** (for 2+ DOF) slice through the tori, revealing their cross-sections as closed curves. When a torus breaks, the neat curve dissolves into a scatter of points — a visual signature of chaos.
- **Lyapunov exponents** (see [chaos theory](chaos.md)) quantify how fast nearby trajectories diverge — zero for motion on a torus, positive for chaos.

The progression from integrable to chaotic is one of the richest subjects in physics. It connects the Hamiltonian mechanics in this document to the chaos analysis tools described in [docs/chaos.md](chaos.md), and it's the reason phase-space thinking — rather than just solving for $q(t)$ — is so powerful.

---

## Contact physics and dissipation

Hamiltonian mechanics conserves energy by construction. Real rigid bodies
that bounce off floors do not. The floor constraint in
`lab/systems/rigid_body/constraints.py` and its JIT counterpart in
`lab/core/rigid_body_jit.py` introduce dissipation through a
sequence of physically motivated impulse models.

### The contact impulse model

When a rigid body's lowest point penetrates the floor ($y < 0$), the
constraint:

1. **Projects** the body upward so the lowest point sits at $y = 0$.
2. **Computes the contact-point velocity** — a combination of the body's
   linear velocity and the cross product $\vec{\omega} \times \vec{r}$
   (the rotational contribution at the contact point):

$$
\vec{v}_c = \frac{\vec{p}}{m} + \vec{\omega}_{\text{world}} \times \vec{r}_{\text{contact}}
$$

where $\vec{r}_{\text{contact}}$ is the vector from the centre of mass to
the lowest point, expressed in the world frame.

3. **Applies a normal impulse** to reverse the normal component of $v_c$:

$$
j_n = -\frac{(1 + e)\, v_{c,n}}{1/m + (\hat{n} \times \vec{r})^T\, I_{\text{world}}^{-1}\, (\hat{n} \times \vec{r})}
$$

This is the **impulse-momentum** equation for a rigid body with rotational
inertia. The denominator accounts for the fact that some of the impulse
goes into changing angular momentum (through the moment arm $\vec{r}$),
not just linear momentum. The coefficient $e$ is the coefficient of
restitution.

### Inertia tensor in the world frame

The inertia tensor $I$ is diagonal in the body frame (by choice of principal
axes). To use it in the impulse equation, we rotate vectors into the body
frame, apply $I^{-1}$, and rotate back:

$$
I_{\text{world}}^{-1}\, \vec{v} = R \, I_{\text{body}}^{-1} \, R^T \, \vec{v}
$$

where $R$ is the rotation matrix corresponding to the body's quaternion
orientation. In practice, we implement this as a sequence of quaternion
rotations rather than constructing the full $3 \times 3$ matrix.

### Coulomb friction

After the normal impulse, the tangential velocity at the contact point
$v_t$ may be nonzero (the body is sliding). Coulomb friction opposes
this sliding with a tangential impulse:

$$
j_t = \min\!\left(\frac{v_t}{1/m + \text{rot. term}},\; \mu \cdot |j_n|\right)
$$

The $\min$ enforces the Coulomb cone: friction force cannot exceed $\mu$
times the normal force.  Below the cone, friction exactly cancels the
sliding (static friction); at the cone limit, the body slides with
kinetic friction.

### Rolling resistance

Pure Coulomb friction is insufficient to stop a convex body from rocking
indefinitely on a flat surface. In reality, contact patches are finite,
and energy is lost to elastic deformation of the surface. We model this
with a **multiplicative angular momentum damping**:

$$
\vec{L} \leftarrow (1 - r_{\text{eff}}) \, \vec{L}
$$

The effective resistance depends on proximity to the body's minimum
resting height:

$$
r_{\text{eff}} = r_0 \cdot \text{clamp}\!\left(1 - \frac{y_{\text{cm}} - y_{\min}}{\delta},\; 0,\; 1\right)
$$

When the body is at its lowest possible centre-of-mass height (lying
flat), $r_{\text{eff}} = r_0$ (full damping). When the body is high
above the floor (e.g., balanced on a corner), $r_{\text{eff}} = 0$ (no
damping — the body can spin freely).

This is a **phenomenological** model, not derived from first principles.
It is designed to produce convergence in finite time while preserving
the qualitative dynamics (which face the body lands on).

For the complete derivation of all contact mechanisms — including dimensional
analysis of thresholds, the snap-to-zero logic, settle detection, and the
code mapping across all three implementations — see
[contact_model.md](contact_model.md).

### Energy flow in a contact event

A single bounce illustrates the energy budget:

$$
KE_{\text{after,normal}} = e^2 \, KE_{\text{before,normal}}
$$

The total energy budget for a single bounce is therefore:

$$
KE_{\text{before}} = KE_{\text{after}} + (1 - e^2)\,KE_{\text{before,normal}} + \Delta E_{\text{friction}} + \Delta E_{\text{rolling}}
$$

At low speeds ($e_{\text{eff}} = 0$), the normal impulse absorbs all
normal KE.  Friction absorbs tangential KE.  Rolling resistance drains
rotational KE.  The body converges to rest in finite time — a necessary
condition for the settle-detection logic described in
[docs/numerical_methods.md](numerical_methods.md).

### Shape-dependent resting states

Different shapes have different stable resting configurations:

| Shape | Dimensions | Resting states | Classification |
|---|---|---|---|
| Coin (US quarter) | $r = 0.01213$ m, $h_t = 0.000875$ m, $m = 0.00567$ kg | Flat on either face, or balanced on edge (unstable) | heads (+1), tails (-1), edge (0) |
| Cube (16 mm die) | $s = 0.016$ m, $m = 0.008$ kg | Flat on any of 6 faces | face index 0–5 |

The lowest point that touches the floor determines the outcome.
For both shapes, the algorithm rotates the body's extreme points
(rim samples for the coin, vertices for the cube) into the world
frame via the quaternion and finds the one with minimum $y$.
See [contact_model.md](contact_model.md) for the full penetration-detection
algorithm and [realistic_parameters.md](realistic_parameters.md) for the
physical constants and their sources.
