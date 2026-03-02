# Contact Model (Floor Constraint)

Deep-dive reference on the rigid-floor contact model used throughout
the drop-simulation codebase.  Covers the physics, the math, the code,
and the numerical tradeoffs.

---

## 1. Physical picture

Drop a coin onto a table from shoulder height.  In the instant of
contact, stresses propagate through the metal, the table deforms
elastically, and momentum is exchanged.  The coin rebounds, spins,
bounces again, and eventually comes to rest.

A fully resolved finite-element simulation would mesh both objects and
integrate stress waves.  That is expensive and unnecessary when all you
want is the final resting orientation.

The model used here is the **impulse approximation**: treat both the body
and the floor as perfectly rigid, and capture all deformation physics
through three scalar coefficients:

| Coefficient        | Symbol | Meaning                                                 |
|--------------------|--------|---------------------------------------------------------|
| Restitution        | $e$    | Ratio of rebound to approach speed (normal direction)   |
| Friction           | $\mu$  | Coulomb coefficient; tangential-to-normal impulse ratio |
| Rolling resistance | $r_r$  | Angular-momentum damping fraction per contact event     |

Contact is instantaneous (zero duration), so there is no contact
force --- only an **impulse** $\vec{J}$ that discretely changes the
body's linear and angular momentum:

$$
\vec{p} \;\to\; \vec{p} + \vec{J}
\qquad\qquad
\vec{L} \;\to\; \vec{L} + \vec{r} \times \vec{J}
$$

where $\vec{r}$ is the lever arm from the centre of mass to the contact
point.  The torque impulse is rotated into the body frame before
application.  The floor is infinitely massive and does not recoil.

---

## 2. Penetration detection

### The lowest-point algorithm

Since our bodies are convex, contact with the plane $y = 0$ occurs at
the body point with the smallest world-frame $y$-coordinate.  We sample
a finite set of **body-frame extreme points**, rotate each into the world
frame via the quaternion sandwich product, and take the minimum.

For a body-frame point $\vec{b}$, the world position is
$\vec{w} = q\,\vec{b}\,q^{*}$, and the surface point's world
$y$-coordinate is $y_{\text{world}} = y_{\text{CM}} + w_y$.
Penetration is $\delta = y_{\text{world, min}}$; contact fires when
$\delta < 0$.

### Coin

Cylinder of radius $R$ = `COIN_RADIUS` = 0.01213 m and half-thickness
$h_t$ = `COIN_HALF_THICK` = 0.000875 m.  Sample points:

- **2 pole points** (face centres): $(0, \pm h_t, 0)$
- **16 rim points**: 8 angles at $k\pi/4$ ($k=0\ldots7$), each at
  $\pm h_t$: $\vec{b}_k^{\pm} = (R\cos(k\pi/4),\;\pm h_t,\;R\sin(k\pi/4))$

Total: **18 sample points**.  The 8-fold angular spacing captures the
true minimum to within $1-\cos(22.5°) \approx 0.08\%$ of the radius.

### Cube

Half-side $h$ = `CUBE_HALF_SIDE` = 0.008 m.  The extreme points are
the **8 vertices** $(\pm h, \pm h, \pm h)$.  No interpolation is
needed: a convex polyhedron's support point in any direction is always
a vertex.

### Code references

- **JIT**: `_lowest_coin()`, `_lowest_cube()` in `lab/core/rigid_body_jit.py`
- **CUDA**: `lowest_point_coin()`, `lowest_point_cube()` in `lab/experiments/drop_gpu.py`
- **OOP**: `RigidBody.lowest_point()` + `_body_frame_extremes()` in `lab/systems/rigid_body/objects.py`

---

## 3. Normal impulse derivation

### Contact-point velocity

Body-frame angular velocity: $\omega_i^{\text{body}} = L_i / I_i$.
Rotate to world frame: $\vec{\omega}^{\text{world}} = q\,\vec{\omega}^{\text{body}}\,q^{*}$.

The velocity at the contact point (lever arm $\vec{r}$ from CM) is

$$
\vec{v}_c = \frac{\vec{p}}{m} + \vec{\omega}^{\text{world}} \times \vec{r}
$$

Normal component (floor normal $\hat{n} = \hat{y}$): $v_n = v_{c,y}$.
Contact response fires only when $v_n < -0.01$ m/s.

### Inertia tensor rotation (body frame to world frame)

The body-frame inertia tensor is diagonal:
$I^{\text{body}} = \operatorname{diag}(I_x, I_y, I_z)$.
Rather than building a $3\times3$ rotation matrix, we use the
**quaternion sandwich** twice:

1. Rotate $(\vec{r} \times \hat{n})$ into the body frame:
   $\vec{u}^{\text{body}} = q^{*}(\vec{r}\times\hat{n})\,q$
2. Apply body-frame inverse inertia:
   $\vec{v}^{\text{body}} = (u_x/I_x,\;u_y/I_y,\;u_z/I_z)$
3. Rotate back: $\vec{v}^{\text{world}} = q\,\vec{v}^{\text{body}}\,q^{*}$

This computes $I_{\text{world}}^{-1}(\vec{r}\times\hat{n}) = R\,I_{\text{body}}^{-1}R^T(\vec{r}\times\hat{n})$ without constructing $R$.

### Effective mass

$$
\frac{1}{m_{\text{eff}}} = \frac{1}{m}
+ \hat{n} \cdot \Bigl[
    \bigl(I_{\text{world}}^{-1} (\vec{r} \times \hat{n})\bigr) \times \vec{r}
  \Bigr]
$$

Since $\hat{n} = \hat{y}$, the dot product picks out the $y$-component.
In code this is the scalar `t_y` (JIT) or `rot_term` (CUDA).

### Velocity-dependent restitution

$$
e_{\text{eff}} =
\begin{cases}
e_0 & \text{if } |v_n| > 0.1 \text{ m/s} \\
0   & \text{otherwise}
\end{cases}
$$

The 0.1 m/s threshold kills **micro-bouncing**: a nearly-resting body
oscillating indefinitely because each tiny bounce returns just enough
energy for the next.  Setting $e=0$ below the threshold makes the
collision perfectly inelastic, letting the damping logic (section 6)
take over.

### Normal impulse magnitude and application

$$
j_n = -\frac{(1 + e_{\text{eff}})\, v_n}{1/m_{\text{eff}}}
$$

Since $v_n < 0$, we get $j_n > 0$ (upward).  Apply:

$$
p_y \;\leftarrow\; p_y + j_n
$$

World-frame torque impulse $\vec{\tau} = \vec{r}\times(j_n\hat{n})$,
rotated into the body frame:

$$
\vec{L} \;\leftarrow\; \vec{L} + q^{*}\,\vec{\tau}\,q
$$

---

## 4. Coulomb friction

### Tangential velocity at the contact point

After applying the normal impulse, recompute the contact velocity with
updated angular momentum.  The tangential component is

$$
\vec{v}_t = \vec{v}_c - (\vec{v}_c \cdot \hat{n})\,\hat{n}
$$

If $|\vec{v}_t| < 10^{-12}$, friction is skipped.

### Tangential effective mass

Unit tangential direction $\hat{t} = \vec{v}_t / |\vec{v}_t|$.  Same
formula as the normal case with $\hat{t}$ replacing $\hat{n}$:

$$
\frac{1}{m_{\text{eff},t}} = \frac{1}{m}
+ \hat{t} \cdot \Bigl[
    \bigl(I_{\text{world}}^{-1} (\vec{r} \times \hat{t})\bigr) \times \vec{r}
  \Bigr]
$$

### Friction cone

The Coulomb model constrains the tangential impulse: $|j_t| \le \mu|j_n|$.

- **Static friction** ($j_{t,\text{desired}} < \mu|j_n|$): exactly
  cancels sliding.  $j_t = |\vec{v}_t| / (1/m_{\text{eff},t})$.
- **Kinetic friction** ($j_{t,\text{desired}} \ge \mu|j_n|$): saturates
  at the cone boundary.  $j_t = \mu|j_n|$.

```python
jt = min(vt_mag / inv_mass_eff_t, friction * abs(j_n))
```

### Application

$$
\vec{p} \;\leftarrow\; \vec{p} - j_t\,\hat{t}
\qquad\qquad
\vec{L} \;\leftarrow\; \vec{L} + q^{*}\bigl(\vec{r}\times(-j_t\hat{t})\bigr)q
$$

---

## 5. Rolling resistance

### Motivation

Even with Coulomb friction, a coin or die can rock quasi-periodically
for thousands of steps.  Real objects lose energy through contact-zone
deformation poorly modelled by point-contact friction.  Rolling
resistance captures this dissipation phenomenologically.

### Near-floor activation

Rolling resistance activates when $\delta < 3 \times \text{char\_size}$,
where `char_size` is `COIN_RADIUS` for coins and `CUBE_HALF_SIDE` for
cubes.

### Stability factor

Let `min_h` be the minimum CM height at rest (`COIN_HALF_THICK` for
coins, `CUBE_HALF_SIDE` for cubes).  The excess height is

$$
\text{excess} = \max(0,\; y_{\text{CM}} - \text{min\_h})
$$

$$
\text{stab} = \max\!\Bigl(0,\; 1 - \frac{\text{excess}}{3 \times \text{char\_size}}\Bigr)
$$

This is a **linear ramp from 1 to 0** over $3\times\text{char\_size}$:
maximum resistance when flat on the floor, zero when bouncing high.
The continuous ramp avoids an abrupt on/off switch at the boundary,
which would cause unphysical energy jumps.  A coin tilted on its rim
has higher excess and receives less damping --- correctly, since rim
contact is more bounce-like than roll-like.

### Angular momentum damping

$$
\vec{L} \;\leftarrow\; (1 - r_r \times \text{stab})\, \vec{L}
$$

Applied to all three body-frame components equally.

---

## 6. Damping and snap-to-zero

### Energy scale

The **characteristic KE scale** is the PE of lifting the body by its
own characteristic size:

$$
\text{ke\_scale} = m \, g \times \text{char\_size}
$$

Total kinetic energy:

$$
KE = \frac{|\vec{p}|^2}{2m}
+ \frac{1}{2}\!\left(\frac{L_x^2}{I_x} + \frac{L_y^2}{I_y} + \frac{L_z^2}{I_z}\right)
$$

### Two-threshold damping (active in near-floor region)

**Zone 1 --- Hard snap-to-zero** (tight threshold):

$$
\text{excess} < 2\times\text{char\_size}
\;\;\text{AND}\;\;
KE < 0.01 \times \text{ke\_scale}
\quad\Longrightarrow\quad
\vec{p} = \vec{0},\;\vec{L} = \vec{0}
$$

Fires when the body is very close to rest and has $<1\%$ of the energy
scale.  This is the **final convergence** mechanism.

**Zone 2 --- Multiplicative damping** (loose threshold):

$$
KE < 0.5 \times \text{ke\_scale}
\quad\Longrightarrow\quad
\vec{p} \leftarrow 0.99\,\vec{p},\;\;
\vec{L} \leftarrow 0.99\,\vec{L}
$$

At 50% of the energy scale, the body is winding down but still
wobbling.  The 0.99 factor gently bleeds energy: over 100 steps it
reduces KE by $\sim 0.99^{200} \approx 0.13\times$.

### Why two thresholds

A single aggressive threshold would either snap too early (visible pop
when a still-rocking body freezes) or snap too late (thousands of
unnecessary steps).  The loose damping accelerates energy decay into
the range where the tight snap fires cleanly.

---

## 7. Settle detection

### KE threshold

$$
\text{ke\_thr} = m \, g \times \text{settle\_h} \times 10^{-4}
$$

where `settle_h` is shape-dependent ($5R$ for coins, $3h$ for cubes).

### Two settle paths

**Path 1 --- KE-based** (100 consecutive steps):

$$
KE < \text{ke\_thr} \;\;\text{AND}\;\; y_{\text{CM}} < \text{settle\_h}
$$

The counter `sc[k]` resets to zero on any violation, so a single energy
spike restarts the countdown.  100 steps at $dt=0.0005$ s = 50 ms.

**Path 2 --- Force-settle timeout** (5000 cumulative steps):

$$
y_{\text{CM}} < \text{settle\_h}
$$

Handles quasi-periodically rocking bodies where KE oscillates above and
below `ke_thr`, perpetually resetting the consecutive counter.  After
5000 steps (2.5 s simulated) near the floor, the body is declared
settled.  The counter resets only if $y > \text{settle\_h}$.

### The half-kick artefact

The settle check **must** be placed between the floor constraint and
the second half-kick.  Otherwise the leapfrog gravity half-kick
reinjects momentum into a zeroed state, creating residual KE of

$$
KE_{\text{residual}} = \frac{m\,g^2\,dt^2}{8}
$$

that can exceed `ke_thr`.  Cross-reference:
[integration.md](integration.md), "The half-kick artefact and settle
detection".

```python
(pos, mom, ori, amom) = _floor(...)       # may zero momenta
ke = translational_ke + rotational_ke     # settle check HERE
if ke < ke_thr and pos_y < settle_h: ...
mom[k, 1] -= mass * g * half_dt           # second half-kick
```

---

## 8. Contact resolution flowchart

```mermaid
flowchart TD
    A["Compute lowest point<br/>(penetration delta)"] --> B{delta < 0?}
    B -- No --> G{Near floor?<br/>delta < 3*char_size}
    B -- Yes --> C["Position correction:<br/>y_CM -= delta"]
    C --> D["Compute contact velocity v_c<br/>v_n = v_c . n"]
    D --> E{v_n < -0.01?}
    E -- No --> G
    E -- Yes --> F["Effective mass + restitution<br/>j_n = -(1+e_eff)*v_n / (1/m_eff)"]
    F --> F3["Apply normal impulse to p, L"]
    F3 --> H{mu > 0 AND |v_t| > eps?}
    H -- No --> G
    H -- Yes --> I["Tangential effective mass<br/>j_t = min(|v_t|/m_eff_t, mu*|j_n|)"]
    I --> J["Apply friction impulse to p, L"]
    J --> G
    G -- No --> Z["Return state"]
    G -- Yes --> K{rr > 0?}
    K -- No --> Z
    K -- Yes --> L["Rolling resistance:<br/>L *= (1 - rr*stab)"]
    L --> M["Compute KE"]
    M --> N{excess < 2*cs<br/>AND KE < 0.01*ke_s?}
    N -- Yes --> O["Hard snap: p=0, L=0"]
    N -- No --> P{KE < 0.5*ke_s?}
    P -- Yes --> Q["Damp: p*=0.99, L*=0.99"]
    P -- No --> Z
    O --> Z
    Q --> Z
```

---

## 9. Code mapping table

Three parallel codepaths are kept in sync manually --- a deliberate
tradeoff for maximum performance in each context.

| Mechanism            | Python OOP                                    | Numba JIT                                   | CUDA                                        |
|----------------------|-----------------------------------------------|---------------------------------------------|---------------------------------------------|
| Lowest point (coin)  | `_body_frame_extremes()` + `lowest_point()`   | `_lowest_coin()`                            | `lowest_point_coin()`                       |
| Lowest point (cube)  | (same)                                        | `_lowest_cube()`                            | `lowest_point_cube()`                       |
| Penetration + impulse| `FloorConstraint._enforce_rigid()`            | `_floor()`                                  | `floor_constraint()`                        |
| Settle detection     | (not in OOP path)                             | `step_bodies()` inner loop                  | `drop_kernel()` inner loop                  |
| Entry point          | `FloorConstraint.enforce(body)`               | `_floor(shape, mass, ...)`                  | `floor_constraint(shape_id, mass, ...)`     |

**Files**:

- Python OOP: `lab/systems/rigid_body/constraints.py` (`FloorConstraint.enforce`)
  and `lab/systems/rigid_body/objects.py` (`RigidBody`)
- JIT: `lab/core/rigid_body_jit.py` (`_floor`, `_lowest_*`, `step_bodies`)
- CUDA: `lab/experiments/drop_gpu.py` (`floor_constraint`, `lowest_point_*`, `drop_kernel`)

### Notable differences between codepaths

- **OOP** uses NumPy arrays and explicit rotation matrices
  (`quat.to_rotation_matrix`).  Its rolling-resistance stability uses a
  fixed 0.05 m threshold rather than shape-dependent `3 * char_size`.
- **JIT** uses scalar Numba functions (`_qr`, `_cross`, etc.) to avoid
  array allocation in the hot loop.
- **CUDA** mirrors JIT almost line-for-line with `@cuda.jit(device=True)`.
  Its settle detection uses a 200-step threshold (vs. 100 in JIT) and
  lacks the 5000-step force-settle timeout.  It also omits the
  multiplicative-damping zone --- only the hard snap fires.

---

## 10. Further reading

- **[integration.md](integration.md)** --- Leapfrog scheme, constraint
  ordering, the half-kick artefact.
- **[numerical_methods.md](numerical_methods.md)** --- Error analysis,
  symplectic properties, adaptive stepping.
- **[physics.md](physics.md)** --- Hamiltonian mechanics, rigid-body
  equations of motion, quaternion conventions.
- **[rotations.md](rotations.md)** --- Quaternion algebra, the
  exponential map, body vs. world frame.
- **[gpu.md](gpu.md)** --- CUDA kernel architecture,
  thread-per-simulation model.
- **[architecture.md](architecture.md)** --- How the OOP, JIT, and CUDA
  codepaths relate.
