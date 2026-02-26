# Deterministic Chaos Theory

*A guide grounded in the coin-toss and double-pendulum experiments from this
project.  Aimed at physics undergraduates who have not yet taken a full
classical-mechanics course.*

---

## 1. Deterministic yet unpredictable

There is a deep philosophical paradox hiding inside our coin-toss simulation.
Open `lab/experiments/drop_experiment.py` and look at the function `drop_body`.
It takes a handful of numbers --- drop height, tilt axis, tilt angle, time step,
restitution coefficient --- and feeds them through Newton's laws with absolutely
no randomness.  There is no call to `random()`.  Every floating-point operation
is prescribed.  Give the function the same inputs, and it will return the same
output, bit for bit, every single time.

The simulation is therefore **deterministic**: the future is uniquely fixed by
the present.  In principle, if you know the initial conditions perfectly, you
know the outcome before the coin hits the ground.

And yet, when you run the parameter sweep and plot the outcome map, the
picture at low heights is orderly --- smooth bands of blue (heads) and red
(tails).  At high heights, the map dissolves into a salt-and-pepper tangle
of colors that *looks* random but is not.

This is the signature of **deterministic chaos**: a system governed by fixed,
known equations whose long-term behaviour is nonetheless practically
unpredictable.  The equations themselves are perfectly knowable; the difficulty
is that any tiny uncertainty in the starting state grows until it overwhelms our
ability to forecast the outcome.

Chaos does not mean "without law."  It means "sensitive to what you don't know."

---

## 2. Sensitivity to initial conditions

### The intuition

Drop a coin from 10 cm with a 45-degree tilt, and it barely rotates before
settling.  Nudge the tilt by a tenth of a degree and the outcome stays the same.
The two nearby initial conditions lead to nearby final states.

Now drop from 3 metres.  The coin tumbles many times on the way down, bounces,
and precesses on the floor.  The same tenth-of-a-degree nudge can flip the
result from heads to tails.  The two initially nearby trajectories have
*diverged*.

### Formal definition

Let $\boldsymbol{x}(t)$ denote the full state of the system (position,
orientation, momenta) at time $t$.  Consider two initial conditions separated by
a small displacement $\boldsymbol{\delta}_0$:

$$
\boldsymbol{x}_2(0) = \boldsymbol{x}_1(0) + \boldsymbol{\delta}_0
$$

If the system is chaotic, the separation grows exponentially:

$$
|\boldsymbol{\delta}(t)| \;\sim\; |\boldsymbol{\delta}_0|\, e^{\lambda t}
$$

The constant $\lambda$ is the **Lyapunov exponent**.  When $\lambda > 0$, the
system is chaotic: nearby trajectories separate at an exponential rate.

### Prediction horizon

Suppose you need the separation to stay below some tolerance $\Delta$ in order
to make a correct prediction (e.g., $\Delta$ is the gap between heads and
tails in orientation space).  Starting from an initial uncertainty
$|\boldsymbol{\delta}_0|$, the time at which you lose predictive power is

$$
t_p \;\sim\; \frac{1}{\lambda}\,\ln\!\left(\frac{\Delta}{|\boldsymbol{\delta}_0|}\right)
$$

This is the **prediction horizon**.  Notice two things:

1. It grows only *logarithmically* with improved measurement precision.
   Reducing your initial error by a factor of 1000 buys you only
   $\ln(1000)/\lambda \approx 6.9/\lambda$ additional time units.
2. It is inversely proportional to $\lambda$.  A more chaotic system
   (larger $\lambda$) gives you less forecasting time.

In the coin experiment, $\lambda$ effectively increases with drop height:
more bounces and more air time allow the exponential divergence to accumulate.
This is why the outcome map is smooth at small heights and fractal at large
heights.

---

## 3. Fractal basins of attraction

### Basins and boundaries

Every coin drop ends in one of three outcomes: heads, tails, or edge.  The set
of initial conditions that lead to a particular outcome is called its **basin of
attraction**.  At low heights, these basins are smooth, simply-connected regions
--- wide bands in the (height, angle) plane.

At high heights, the basins develop an increasingly intricate boundary.  Zoom
into the boundary between the blue (heads) and red (tails) regions of the
outcome map and you will find that the boundary is not a clean curve.  Instead,
thin filaments of one color penetrate deep into the other's territory, and those
filaments themselves contain sub-filaments, and so on.

### What makes a boundary fractal

A fractal boundary has **self-similarity across scales**: the pattern you see at
one magnification reappears (statistically, at least) when you zoom in further.
Formally, the boundary has a non-integer **fractal dimension** $d_f$ satisfying

$$
1 < d_f < 2
$$

for our two-dimensional parameter space.  A smooth curve has $d_f = 1$.
A space-filling curve has $d_f = 2$.  The basin boundary sits between the two.

### Why it happens

The coin's dynamics define a map from initial conditions to outcomes.  At the
boundary between basins, infinitesimally close initial conditions map to
different outcomes.  Because the dynamics stretch and fold phase space (the
hallmark of chaos), the pre-images of the basin boundary form an intricate web.
In mathematical terms, the boundary is an invariant set of zero Lebesgue measure
but with Hausdorff dimension strictly greater than one.

### Connection to the Mandelbrot set

The Mandelbrot set arises from iterating $z \mapsto z^2 + c$ in the complex
plane and colouring each $c$ by whether the orbit escapes.  Its boundary is
a fractal of dimension 2.  Our outcome map uses a different dynamical system
--- rigid-body mechanics instead of complex iteration --- but the mechanism is
analogous: a deterministic map from parameters to outcomes produces fractal
boundaries whenever the dynamics are chaotic.  The coin-toss outcome map is,
in a real sense, a physicist's Mandelbrot set.

---

## 4. The double pendulum

### The canonical example

Open `lab/systems/pendulums.py`.  The function `double_pendulum` constructs the
exact (non-linearised) Hamiltonian for two point masses on rigid, massless rods.
The generalised coordinates are the two angles $\theta_1$ and $\theta_2$, and
the conjugate momenta are $p_1$ and $p_2$.

The Hamiltonian is

$$
H = T(q, p) + V(q)
$$

where the potential energy is

$$
V = -(m_1 + m_2)\,g\,l_1\cos\theta_1 \;-\; m_2\,g\,l_2\cos\theta_2
$$

and the kinetic energy has the complicated coupling term visible in the source:
it depends on $\cos(\theta_1 - \theta_2)$, which mixes the two degrees of
freedom nonlinearly.

### Why two degrees of freedom matter

A one-degree-of-freedom Hamiltonian system (like the simple pendulum in the same
file) cannot be chaotic.  Its phase space is two-dimensional, and energy
conservation confines trajectories to one-dimensional curves --- they have
nowhere to wander.

With two degrees of freedom the phase space is four-dimensional.  Energy
conservation restricts motion to a three-dimensional surface.  That leaves
enough room for trajectories to diverge exponentially from their neighbours.

This is the content of the **KAM theorem** (Kolmogorov--Arnold--Moser, 1954--63).
At low energies, the double pendulum oscillates quasi-periodically: its
trajectories live on two-dimensional tori embedded in the three-dimensional
energy surface.  As the energy increases, those tori break up, and a "chaotic
sea" floods the phase space.

### Poincare sections

A Poincare section is a way to visualise four-dimensional dynamics on a
two-dimensional plot.  You pick a surface of section --- for example, the plane
$\theta_2 = 0$ with $\dot{\theta}_2 > 0$ --- and record $(\theta_1, p_1)$
every time the trajectory crosses it.

- **Regular orbits** appear as smooth closed curves (the intersection of a torus
  with the section plane).
- **Chaotic orbits** fill a scattered cloud (the "chaotic sea").

At the transition energy, you see islands of regularity surrounded by a sea of
chaos --- one of the most beautiful images in all of physics.

---

## 5. Lyapunov exponents in practice

### Numerical algorithm

The definition $|\boldsymbol{\delta}(t)| \sim |\boldsymbol{\delta}_0|\,
e^{\lambda t}$ suggests a straightforward numerical recipe:

1. Choose a reference trajectory $\boldsymbol{x}(t)$ and a perturbed trajectory
   $\boldsymbol{x}'(t) = \boldsymbol{x}(t) + \boldsymbol{\delta}(t)$.
2. Evolve both forward by a short interval $\tau$.
3. Measure the growth factor
   $r = |\boldsymbol{\delta}(t + \tau)| \,/\, |\boldsymbol{\delta}(t)|$.
4. **Renormalise**: rescale $\boldsymbol{\delta}$ back to its original
   magnitude, preserving its direction.  This prevents the separation from
   saturating (it cannot grow larger than the size of phase space).
5. Accumulate $\ln r$ and repeat.

After $N$ intervals the maximal Lyapunov exponent is estimated as

$$
\lambda_{\max} \;\approx\; \frac{1}{N\tau}\sum_{k=1}^{N}\ln r_k
$$

### Spectrum of exponents

A system with $n$ degrees of freedom has $2n$ Lyapunov exponents (one for each
phase-space direction).  For a Hamiltonian system, they come in pairs
$(\lambda, -\lambda)$ (a consequence of Liouville's theorem: phase-space volume
is conserved).  In addition, every conserved quantity contributes a zero
exponent.  So a Hamiltonian system with one conserved energy has the exponent
structure:

$$
\lambda_1 \geq 0 = 0 \geq -\lambda_1
$$

for one degree of freedom (always non-chaotic), and

$$
\lambda_1 \geq \lambda_2 \geq 0 = 0 \geq -\lambda_2 \geq -\lambda_1
$$

for two degrees of freedom, where $\lambda_1 > 0$ signals chaos.

### Future direction

Adding a Lyapunov-exponent computation to the project would be a natural next
step.  One could evolve the double pendulum from `lab/systems/pendulums.py`
using the existing symplectic integrators, track a tangent vector alongside the
trajectory, and renormalise periodically to estimate $\lambda_{\max}$ as a
function of energy.

---

## 6. Practical implications

### Weather prediction

In 1963, Edward Lorenz discovered chaos while modelling atmospheric convection.
His three-variable system --- the famous Lorenz attractor --- has a maximal
Lyapunov exponent that limits weather prediction to roughly 10--14 days, no
matter how many sensors you deploy.  The prediction horizon formula

$$
t_p \sim \frac{1}{\lambda}\ln\!\left(\frac{\Delta}{|\boldsymbol{\delta}_0|}\right)
$$

explains why: improving initial-condition accuracy by a factor of 10 buys you
only $\ln(10)/\lambda \approx 2.3/\lambda$ extra days.

### The three-body problem

Newton solved the two-body gravitational problem exactly.  Add a third body and
the system generally becomes chaotic --- Poincare proved in 1890 that no
closed-form solution exists.  Astrophysical three-body encounters must be
treated statistically because individual outcomes are sensitive to initial
conditions at the level of floating-point precision.

### Molecular dynamics

Simulations of $10^3$--$10^6$ interacting atoms have enormous Lyapunov
exponents.  Individual trajectories diverge in picoseconds, but statistical
averages --- temperature, pressure, diffusion coefficients --- converge because
they are insensitive to the precise trajectory.  This is the bridge between
chaos and statistical mechanics (Section 7).

### Finite precision as a fundamental limit

Every computer represents real numbers with finite precision.  IEEE 754
double-precision floats carry about 16 significant decimal digits, which means
the initial condition is known to a relative accuracy of roughly $10^{-16}$.
For a system with Lyapunov exponent $\lambda$, the prediction horizon due to
floating-point rounding alone is

$$
t_{\text{float}} \;\sim\; \frac{16\,\ln 10}{\lambda} \;\approx\; \frac{37}{\lambda}
$$

Beyond this time, even a simulation with *perfect physics* and *exact initial
conditions* (up to double precision) will diverge from the true trajectory.

---

## 7. Connection to statistical mechanics

### The ergodic hypothesis

Statistical mechanics rests on a postulate: over long times, a closed
Hamiltonian system visits all regions of its energy surface with equal
probability.  This is the **ergodic hypothesis**, and it lets us replace a
time average (what we measure in an experiment) with an ensemble average
(what we compute in theory):

$$
\langle A \rangle_{\text{time}} = \lim_{T \to \infty}\frac{1}{T}\int_0^T A\bigl(\boldsymbol{x}(t)\bigr)\,dt
\;=\;
\int A(\boldsymbol{x})\,\rho(\boldsymbol{x})\,d\boldsymbol{x}
\;=\; \langle A \rangle_{\text{ensemble}}
$$

where $\rho$ is the microcanonical distribution (uniform on the energy surface).

### Chaos makes ergodicity plausible

Why should a deterministic trajectory visit all of phase space?  The answer is
chaos.  Exponential sensitivity means that trajectories stretch and fold
through phase space, mixing it like a baker kneads dough.  Without chaos,
trajectories are confined to low-dimensional tori and ergodicity fails.  In the
double pendulum, the transition from regular to chaotic motion as energy
increases mirrors the transition from non-ergodic to ergodic behaviour.

### From Newton to Boltzmann

Deterministic chaos thus provides the conceptual bridge between Newtonian
mechanics and thermodynamics:

1. **Newton**: every particle obeys $F = ma$.  The future is determined.
2. **Chaos**: with many degrees of freedom, exponential sensitivity makes
   individual trajectories unpredictable in practice.
3. **Boltzmann**: replace trajectory-level prediction with probabilistic
   statements.  Entropy increases because the system explores phase space
   ergodically, and the overwhelming majority of phase-space volume corresponds
   to the equilibrium macrostate.

This is not hand-waving --- it is the deep reason that thermodynamics works even
though the underlying mechanics is deterministic and time-reversible.

---

## 8. What our experiments show

### GPU vs CPU: same physics, different answers

Run the coin-toss experiment on the CPU:

```python
from lab.experiments.drop_experiment import sweep_drop
results_cpu = sweep_drop("coin", heights, angles, tilt_axis="x")
```

Then run the same sweep on the GPU:

```python
from lab.experiments.drop_gpu import sweep_drop_gpu
results_gpu = sweep_drop_gpu("coin", heights, angles, tilt_axis="x")
```

Both implementations solve the same equations: rigid-body dynamics with gravity,
floor collisions, Coulomb friction, and rolling resistance.  The integrator
is the same leapfrog (symplectic) scheme with the same time step `dt=0.001`.
The physical parameters are identical.

And yet, at high drop heights, the two result matrices **disagree** on a
significant fraction of grid points.

### Why the disagreement is not a bug

The CPU code in `drop_body` (inside `lab/experiments/drop_experiment.py`)
executes floating-point operations sequentially via NumPy.  The GPU kernel
in `lab/experiments/drop_gpu.py` runs the same mathematics on a CUDA core, but
with different instruction scheduling.  IEEE 754 floating-point arithmetic is
not associative:

$$
(a + b) + c \;\neq\; a + (b + c)
$$

in general, because each addition rounds to the nearest representable float.
A GPU fused-multiply-add computes $a \times b + c$ in one instruction with one
rounding step; the CPU may use separate multiply and add with two rounding
steps.

These differences are of order $10^{-16}$.  At low drop heights, the outcome
basins are wide and a $10^{-16}$ perturbation does not cross a boundary.  At
high drop heights, the basin boundaries are fractal, and *any* perturbation,
no matter how small, can land on the wrong side.

### This IS chaos in action

The CPU/GPU discrepancy is not a numerical error to be fixed.  It is a physical
demonstration of deterministic chaos:

1. The two computations start from the same (double-precision) initial
   conditions.
2. They implement the same physical laws.
3. Hardware-level differences in floating-point rounding introduce an effective
   initial perturbation of magnitude $\sim 10^{-16}$.
4. The chaotic dynamics amplify that perturbation exponentially: after enough
   bounces, it exceeds the basin width, and the two computations disagree.

The prediction-horizon formula tells us exactly when this happens:

$$
t_{\text{diverge}} \;\sim\; \frac{1}{\lambda}\,\ln\!\left(\frac{\Delta}{10^{-16}}\right)
$$

For high drop heights (long simulation times, many bounces), $t_{\text{diverge}}$
falls within the simulation window, and the outcomes split.

### Visualising the disagreement

Subtracting the two outcome maps produces a "disagreement mask" that is nearly
empty at low heights and dense at high heights.  The mask traces out the
fractal basin boundaries --- an empirical map of where chaos lives in
parameter space.

---

## 9. Watching chaos unfold: the live dashboard

The static outcome map — a colour-coded image where each pixel represents a
completed simulation — is a post-hoc view of the basin structure.  The **live
dashboard** (`--live` flag) adds a temporal dimension: you watch the chaos
*form* in real time.

### What you see

The dashboard has three panels:

1. **3D scatter** — every grid point is a dot, spatially arranged by
   (angle, height).  All dots start at their respective drop heights and
   fall simultaneously.  You see a "rain" of objects settling onto the
   floor.
2. **Outcome map** — starts blank (gray) and fills in pixel by pixel as
   objects settle.  Low-height regions fill first (short drops settle
   quickly); high-height regions fill last.
3. **Histogram** — bar chart of outcome counts, growing as the map fills.

### What chaos looks like in real time

At low heights (the bottom rows of the map), objects settle within a
fraction of a second.  The colour pattern that appears is smooth —
clean bands of blue (heads) and red (tails) for the coin, ordered
patches of face colours for the cube.  These are the regular basins.

At high heights (the top rows), objects bounce and tumble for seconds.
They remain gray dots in the scatter view while the lower rows have
long since coloured in.  When they finally settle, the colours appear
in a seemingly random salt-and-pepper pattern — adjacent pixels land
on different outcomes.

The transition is not gradual.  There is a critical height range where
the map shifts from smooth to fractal.  Watching the live dashboard
makes this transition **viscerally obvious**: the lower half fills
quickly and neatly; the upper half stutters in, pixel by pixel, with
no discernible pattern.

### The connection to prediction horizons

The time it takes a dot to change from gray to coloured is proportional
to the simulation time needed for that (height, angle) pair to settle.
Longer simulation times mean more bounces, more opportunities for
exponential divergence, and therefore finer-grained basin boundaries.

The live dashboard thus provides a direct visual mapping of the prediction
horizon $t_p$ onto the parameter space: regions where dots linger are
regions where $t_p$ is comparable to the drop time — the edge of
predictability.

### Quantitative observation

Run the live dashboard for the coin with a fine grid:

```bash
python experiments/drop_coin.py --live --nh 20 --na 30
```

Watch the 3D panel.  You will notice:
- At $h < 0.5$ m, all objects settle within ~50 frames (~1.5 s animation time).
- At $h \approx 1.5$ m, some objects are still rocking after 200 frames.
- At $h > 2.5$ m, a few objects linger for 500+ frames — these are the points
  sitting on fractal basin boundaries where the outcome is hypersensitive to
  initial conditions.

The histogram panel reveals the statistical structure: despite the chaotic
boundary, the *bulk* distribution of heads vs. tails converges quickly.
Most of the parameter space is solidly inside one basin.  Chaos lives at the
boundaries, not in the bulk — echoing the connection between chaos and
statistical mechanics from Section 7.

---

## Summary of key ideas

| Concept | One-sentence definition |
|---|---|
| Deterministic chaos | Deterministic dynamics that are practically unpredictable due to sensitivity to initial conditions. |
| Lyapunov exponent $\lambda$ | The exponential rate at which nearby trajectories diverge; $\lambda > 0$ means chaos. |
| Prediction horizon $t_p$ | The time beyond which forecasts are unreliable: $t_p \sim (1/\lambda)\ln(\Delta/\delta_0)$. |
| Basin of attraction | The set of initial conditions that lead to a given outcome. |
| Fractal basin boundary | A boundary between basins with non-integer dimension, exhibiting self-similarity at all scales. |
| KAM theorem | Describes the transition from regular to chaotic motion in Hamiltonian systems as a perturbation increases. |
| Poincare section | A lower-dimensional slice through phase space that reveals the structure of orbits. |
| Ergodic hypothesis | Over long times, a trajectory visits all accessible phase space uniformly, connecting mechanics to thermodynamics. |

---

## Further reading

- Strogatz, S. H. *Nonlinear Dynamics and Chaos* (2nd ed., 2015). The standard
  undergraduate textbook; Chapters 9--12 cover everything here in depth.
- Lorenz, E. N. "Deterministic Nonperiodic Flow." *J. Atmos. Sci.* **20**,
  130 (1963). The paper that launched chaos theory.
- Ott, E. *Chaos in Dynamical Systems* (2nd ed., 2002). Excellent treatment of
  fractal basins and Lyapunov exponents.
