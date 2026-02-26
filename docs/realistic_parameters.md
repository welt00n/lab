# Realistic Physical Parameters

Every physical constant, derived quantity, and dimensioned threshold in
the simulation.  All values SI.  Cross-references:
[contact_model.md](contact_model.md), [integration.md](integration.md),
[numerical_methods.md](numerical_methods.md).

---

## 1. The objects

### US quarter (coin)

| Property | Value | Source |
|----------|-------|--------|
| Diameter | 24.26 mm | US Mint spec (31 USC 5112) |
| Radius $r$ | 0.01213 m | |
| Thickness $t$ | 1.75 mm | US Mint spec |
| Half-thickness $h_t$ | 0.000875 m | |
| Mass $m$ | 5.67 g (0.00567 kg) | US Mint spec |
| Material | Cupronickel (75 Cu / 25 Ni clad over Cu core) | |
| Aspect ratio $t/d$ | 0.0721 | |

### Standard 16 mm casino die (cube)

| Property | Value | Source |
|----------|-------|--------|
| Side $s$ | 16 mm | Precision hobby die spec |
| Half-side $a$ | 0.008 m | |
| Mass $m$ | 8 g (0.008 kg) | Measured (acrylic + pip fill) |
| Material | Acrylic (PMMA), $\rho \approx 1190\;\text{kg/m}^3$ | |

The 8 g mass exceeds $\rho s^3 \approx 4.87\;\text{g}$ because casino
dice use denser pip fills.  The value is a representative measured weight.

In code (`live_dashboard.py`):

```python
COIN_RADIUS, COIN_HALF_THICK, COIN_MASS = 0.01213, 0.000875, 0.00567
CUBE_HALF_SIDE, CUBE_MASS               = 0.008,   0.008
```

---

## 2. Inertia tensor derivations

Body frame, origin at centre of mass.

### Thin disc (coin)

The thin-disc approximation neglects $h_t$; relative error $\sim (h_t/r)^2 \approx 0.5\%$.

**Axial moment** $I_{\text{axis}}$ (symmetry axis, $y$ in body frame):

$$
I_{\text{axis}}
= \int_0^{2\pi}\!\!\int_0^r \sigma\,\rho^3\,d\rho\,d\theta
= \frac{m}{\pi r^2}\cdot\pi\cdot\frac{r^4}{2}
= \frac{mr^2}{2}
$$

**Diametral moment** $I_{\text{diam}}$ (any axis in the disc plane).
By the perpendicular-axis theorem $I_x + I_z = I_y$ and symmetry $I_x = I_z$:

$$
I_{\text{diam}} = \frac{I_{\text{axis}}}{2} = \frac{mr^2}{4}
$$

Direct integral confirms:

$$
I_{\text{diam}}
= \sigma\!\int_0^r \rho^3 d\rho \int_0^{2\pi}\!\sin^2\!\theta\,d\theta
= \frac{m}{\pi r^2}\cdot\frac{r^4}{4}\cdot\pi
= \frac{mr^2}{4}
$$

**US quarter numerical values:**
$I_{\text{diam}} = 0.00567 \times 0.25 \times 0.01213^2 = 2.086 \times 10^{-7}\;\text{kg\,m}^2$,
$I_{\text{axis}} = 4.172 \times 10^{-7}\;\text{kg\,m}^2$.

**In code** (`_get_inertia`, `live_dashboard.py:341`):

```python
m, r = COIN_MASS, COIN_RADIUS
return (m*0.25*r*r, m*0.50*r*r, m*0.25*r*r)   # Ix, Iy, Iz
```

OOP path (`objects.py:171`): `I_diam = mass * radius**2 / 4.0`, `I_axis = mass * radius**2 / 2.0`.

### Uniform cube (die)

All three principal moments identical by symmetry.

$$
I_{\text{face}}
= \frac{m}{s^3}\int_{-a}^{a}\!\!\int_{-a}^{a}\!\!\int_{-a}^{a}
  (y^2+z^2)\,dx\,dy\,dz
= \frac{m}{s^3}\cdot s\cdot\frac{8a^4}{3}
= \frac{ms^2}{6}
$$

**16 mm die:** $I_{\text{face}} = 0.008 \times 0.016^2 / 6 = 3.413 \times 10^{-7}\;\text{kg\,m}^2$.

**In code** (`live_dashboard.py:345`):

```python
m, s = CUBE_MASS, 2.0 * CUBE_HALF_SIDE
I = m * s * s / 6.0
return (I, I, I)
```

### Verification

| Object | Formula | Numerical value (kg m^2) |
|--------|---------|--------------------------|
| Coin $I_x = I_z$ | $mr^2/4$ | $2.086 \times 10^{-7}$ |
| Coin $I_y$ | $mr^2/2$ | $4.172 \times 10^{-7}$ |
| Cube $I$ | $ms^2/6$ | $3.413 \times 10^{-7}$ |

All three code paths (Numba JIT, CUDA in `drop_gpu.py`, OOP in `objects.py`) produce identical values.

---

## 3. Scale invariance

**Free fall is mass-independent.** $a = g$ regardless of $m$.

**Contact angular velocity scales as $1/r$.** Consider a rim impact.
The normal impulse (see [contact_model.md](contact_model.md)) is

$$
j_n = \frac{-(1+e)\,v_n}{1/m + (\mathbf{r}_c\times\hat{n})\cdot I^{-1}(\mathbf{r}_c\times\hat{n})}
$$

With $I \sim mr^2$ and $|\mathbf{r}_c| \sim r$, the denominator scales as
$1/m + r^2/(mr^2) = 2/m$, so $j_n \sim mv$.  The torque impulse
$|\Delta L| = |j_n||\mathbf{r}_c| \sim mvr$, giving

$$
\omega \sim \frac{|\Delta L|}{I} \sim \frac{mvr}{mr^2} = \frac{v}{r}
$$

Mass cancels entirely.

**Aspect ratio is the only shape parameter.** For a disc,
$I_{\text{diam}}/I_{\text{axis}} = 1/2$ regardless of $m$ or $r$.  The
dimensionless ratio $\alpha = t/d$ is the sole geometric parameter
governing coin-flip outcome statistics.  A US quarter ($\alpha = 0.0721$)
and any geometrically similar disc produce statistically identical outcome
maps at the same drop heights and initial angles.

---

## 4. Dimensional analysis for thresholds

### Natural energy scale

$$
E_0 = m\,g\,L
$$

where $L$ = `COIN_RADIUS` (coins) or `CUBE_HALF_SIDE` (cubes) --- the PE
of lifting the object by its own characteristic size.

In code (`live_dashboard.py:310`): `ke_scale = mass * 9.81 * char_size`.

Quarter: $E_0 = 6.74 \times 10^{-4}\;\text{J}$.  Die: $E_0 = 6.28 \times 10^{-4}\;\text{J}$.

### Threshold table

| Threshold | Value | Dimensional origin |
|-----------|-------|--------------------|
| `ke_thr` (settle) | $m g h_{\text{settle}} \times 10^{-4}$ | 0.01% of PE at settle height |
| KE snap-to-zero | $0.01 \times E_0$ | 1% of one-body-length PE |
| KE damp zone | $0.5 \times E_0$ | 50% of one-body-length PE |
| `near_floor` (coin) | $3 \times$ `COIN_RADIUS` | ~3 body radii above floor |
| `near_floor` (cube) | $3 \times$ `CUBE_HALF_SIDE` | ~3 half-sides above floor |
| Rolling resistance range | $3 \times$ `char_size` | Same as `near_floor` |
| $v_n$ restitution cutoff | 0.1 m/s | Empirical; below this, micro-bouncing is unphysical |
| Consecutive steps (KE) | 100 | At $dt=0.0005$, this is 0.05 s |
| Force-settle timeout | 5000 steps | At $dt=0.0005$, this is 2.5 s |

### Tradeoffs

**`ke_thr`** scales with drop height via $h_{\text{settle}}$.  The
$10^{-4}$ factor is small enough that the object is visually stationary,
large enough to avoid false positives from floating-point noise.

**KE snap-to-zero** ($0.01\,E_0$): residual velocity at this KE is well
below the 0.1 m/s restitution cutoff, preventing indefinite sub-pixel
vibration.  Activates only when the object is within $2L$ of rest height.

**KE damp zone** ($0.5\,E_0$, factor 0.99/step): half-life ~0.035 s at
$dt = 0.0005$.  Mimics acoustic radiation and internal dissipation during
the final settling.  Above $0.5\,E_0$ the dynamics are purely Hamiltonian.

**Restitution cutoff** (0.1 m/s): below this speed, surface roughness
dominates and the rigid-body model cannot capture the physics.  Setting
$e = 0$ avoids unphysical rattle.  See
[contact_model.md](contact_model.md).

**Force-settle timeout** (5000 steps = 2.5 s): catches slow rolling where
KE never drops below `ke_thr`.  See
[numerical_methods.md](numerical_methods.md) for the settle algorithm.

---

## 5. Timestep selection

### Worst-case angular velocity

From Section 3, with $I_{\text{diam}} = mr^2/4$:

$$
\omega_{\max} = \frac{j_n \cdot r}{I_{\text{diam}}} = \frac{mv \cdot r}{mr^2/4} = \frac{4v}{r}
$$

For a US quarter from $h = 2$ m: $v = \sqrt{2gh} \approx 6.26$ m/s,

$$
\omega_{\max} \approx \frac{4 \times 6.26}{0.01213} \approx 2064\;\text{rad/s}
$$

### Rotation per step

| $dt$ (s) | Steps / 2 s | Worst-case $\Delta\theta$ | Status |
|-----------|-------------|---------------------------|--------|
| 0.00025 | 8000 | ~29 deg | Comfortable |
| **0.0005** | **4000** | **~59 deg** | **Default** |
| 0.001 | 2000 | ~118 deg | Unstable for edge impacts |
| 0.002 | 1000 | ~236 deg | Fails catastrophically |

The heuristic stability limit for quaternion exponential-map integration
is ~30 deg/step (see [integration.md](integration.md)).  At the default
$dt = 0.0005$, the worst case (~59 deg) exceeds this but occurs only for
near-edge rim impacts --- a measure-zero slice of initial conditions.
For the vast majority of geometries the effective moment arm is shorter
than $r$, keeping $\omega$ well below $\omega_{\max}$.

At $dt = 0.001$ the worst case reaches ~118 deg, causing energy blow-up
and quaternion denormalisation.  The default $dt = 0.0005$ is the
pragmatic choice: halving to 0.00025 doubles the cost with negligible
improvement in outcome statistics.

---

## 6. Visual vs physical mesh

### Physical dimensions (all dynamics)

All physics --- gravity, impulse, inertia, contact detection --- uses the
real dimensions from Section 1.  `_lowest_coin` rotates face centres
(at $\pm h_t$ along body-$y$) and 8 rim points by the quaternion;
`_lowest_cube` rotates all 8 vertices.  Both test against $y = 0$.

### Display dimensions (cosmetic)

The real objects are nearly invisible at normal camera distances.  The
display mesh is enlarged to a uniform 0.15 m:

```python
VIS_COIN_RADIUS = 0.15    # 0.15 / 0.01213 ≈ 12.4×
VIS_COIN_HT     = 0.01    # 0.01 / 0.000875 ≈ 11.4×
VIS_CUBE_HALF   = 0.15    # 0.15 / 0.008   ≈ 18.75×
```

The coin and cube scale factors differ (12.4x vs 18.75x) because both map
to the same visual size for uniform viewport appearance.

### The `user_matrix` transform

PyVista `DropScene` (`pyvista_scene.py`) and the matplotlib
`Poly3DCollection` path both:

1. Create the mesh once at visual scale (`pv.Cylinder` / `pv.Box`).
2. Each frame, `build_user_matrix(qw, qx, qy, qz, px, py, pz)` produces
   a 4x4 affine encoding the physics rotation + translation.
3. VTK's `user_matrix` applies this to the visual mesh.

The physics-to-display coordinate swap (y-up $\to$ z-up) is baked into
`build_user_matrix`: $\text{display} = (x_{\text{phys}},\, z_{\text{phys}},\, y_{\text{phys}})$,
with the rotation conjugated by the permutation matrix $S$.

### Why the separation matters

Mixing scales would either make objects invisible (physical scale for
display) or produce wrong physics (visual scale for dynamics --- a 0.15 m
"coin" at 5.67 g has aerogel density).  The separation ensures outcome
statistics are governed by real parameters while the viewport remains
legible.  Changing display scale requires editing only the `VIS_*`
constants; zero physics code is touched.

---

## References

- US Mint coin specifications: 31 USC 5112; [usmint.gov](https://www.usmint.gov/learn/coin-and-medal-programs/coin-specifications)
- Casino die dimensions: 19.05 mm (3/4") for regulation; 16 mm is the standard precision hobby size
- Cupronickel: ASM Handbook, Vol. 2
- PMMA density: ~1190 kg/m^3 (MatWeb)
