# Rotations: From Scratch to Quaternions

**Prerequisite**: You know what a 3D coordinate system is (x, y, z axes).
No classical mechanics required.

## The problem

A point particle only has a position: where is it?

A rigid body also has an **orientation**: which way is it facing?
If you drop a cube, it tumbles. We need a way to track "how is it
rotated right now?" at every timestep.

There are three common representations. All describe the same physical
thing — they're different *encodings* of "how the object's body axes
are rotated relative to the world axes."

---

## Option 1: Euler angles

The idea you're recalling. Three angles, usually called
$(\phi, \theta, \psi)$ or (yaw, pitch, roll):

1. Rotate by $\phi$ around the z-axis
2. Then rotate by $\theta$ around the (new) y-axis
3. Then rotate by $\psi$ around the (new) x-axis

**Pros**: Easy to picture. Only 3 numbers.

**Fatal flaw — gimbal lock**: When the middle angle $\theta$ reaches
$\pm 90°$, the first and third rotations become the *same* rotation.
You lose a degree of freedom. The math blows up — derivatives become
infinite and your integrator crashes.

This is not a theoretical edge case. A tumbling cube regularly passes
through these orientations.

> Gimbal lock is what killed Euler angles for Apollo spacecraft
> navigation. NASA switched to quaternions.

**Verdict**: Not usable for a physics engine.

---

## Option 2: Rotation matrices

A $3 \times 3$ matrix $R$ that rotates any vector from body frame to
world frame:

$$
\vec{v}_{\text{world}} = R \; \vec{v}_{\text{body}}
$$

**Pros**: No gimbal lock. Rotating a vector is just matrix multiplication.

**Cons**: 9 numbers to store instead of 3. And they **drift**: after
thousands of small updates, $R$ is no longer a valid rotation matrix
(it should satisfy $R^T R = I$ and $\det R = 1$). Re-orthogonalising
a $3 \times 3$ matrix every step is expensive.

**Verdict**: Works, but wasteful. Used in some engines, but there's a
better option.

---

## Option 3: Quaternions (what we use)

### What is a quaternion, really?

Forget the abstract algebra. Operationally, a quaternion is
**4 numbers that encode a rotation**:

$$
q = (w, x, y, z)
$$

The rotation it represents is:

> "Rotate by angle $\theta$ around the axis $(a_x, a_y, a_z)$."

And the encoding is:

$$
w = \cos\frac{\theta}{2}, \quad x = a_x \sin\frac{\theta}{2}, \quad y = a_y \sin\frac{\theta}{2}, \quad z = a_z \sin\frac{\theta}{2}
$$

That's it. Four numbers. One angle, one axis.

### Examples

| Rotation | Quaternion $(w, x, y, z)$ |
|---|---|
| No rotation (identity) | $(1, 0, 0, 0)$ |
| 90° around y-axis | $(\cos 45°, 0, \sin 45°, 0) = (0.707, 0, 0.707, 0)$ |
| 180° around z-axis | $(\cos 90°, 0, 0, \sin 90°) = (0, 0, 0, 1)$ |

### Why the half-angle?

Two quaternions that differ only in sign ($q$ and $-q$) represent the
**same** rotation. The half-angle formula makes composition (combining
rotations) work out cleanly with simple multiplication rules.

### The unit constraint

A valid rotation quaternion must have length 1:

$$
w^2 + x^2 + y^2 + z^2 = 1
$$

This means quaternions live on the surface of a 4D unit sphere.
After each timestep, we **normalise** the quaternion (divide by its
length) to stay on this sphere. This is one division — much cheaper
than re-orthogonalising a $3 \times 3$ matrix.

---

## The three operations we need

### 1. Rotate a vector

Given a quaternion $q$ and a vector $\vec{v}$, the rotated vector is:

$$
\vec{v}' = q \otimes (0, \vec{v}) \otimes q^*
$$

where $q^* = (w, -x, -y, -z)$ is the conjugate. This is the
"sandwich product." In code:

```python
from lab.core.quaternion import rotate_vector
v_world = rotate_vector(orientation, v_body)
```

### 2. Combine two rotations

To apply rotation $q_1$ then $q_2$, multiply the quaternions:

$$
q_{\text{total}} = q_2 \otimes q_1
$$

Order matters — rotations don't commute. In code:

```python
from lab.core.quaternion import multiply
combined = multiply(q2, q1)
```

### 3. Small rotation from angular velocity

If the body is spinning with angular velocity $\vec{\omega}$ for a
small time $dt$, the rotation that happened during that time is:

$$
\Delta q = \left(\cos\frac{\theta}{2}, \; \sin\frac{\theta}{2} \; \hat{\omega}\right) \quad \text{where } \theta = |\vec{\omega}| \cdot dt
$$

This is the "exponential map" — it converts a spin rate into a
rotation quaternion. In code:

```python
from lab.core.quaternion import exp_map
dq = exp_map(omega, dt)
```

The integrator uses this every timestep to update the body's
orientation.

---

## Comparison summary

| | Euler angles | Rotation matrix | Quaternion |
|---|---|---|---|
| Numbers stored | 3 | 9 | 4 |
| Gimbal lock? | **Yes** | No | No |
| Drift correction | N/A (breaks first) | Expensive | Cheap (normalise) |
| Compose rotations | Messy trig | Matrix multiply | Quaternion multiply |
| Used in this engine | No | Derived when needed | **Primary representation** |

---

## Where quaternions appear in the codebase

| File | What it does with quaternions |
|---|---|
| `lab/core/quaternion.py` | All the math: multiply, conjugate, rotate_vector, exp_map, normalise, to_rotation_matrix |
| `lab/systems/rigid_body/objects.py` | `RigidBody.orientation` is a quaternion `[w, x, y, z]`. `lowest_point()` and `mesh()` use `rotate_vector` to transform body-frame geometry to world frame |
| `lab/systems/rigid_body/world.py` | The integrator calls `exp_map` to update orientation each timestep, then normalises |
| `lab/systems/rigid_body/constraints.py` | Floor collision rotates vectors between body and world frame to compute the contact impulse |

---

## Euler angles vs quaternions, visually

Think of Euler angles as "turn left, then tilt up, then roll." Three
separate sequential rotations. The problem is that the second rotation
can align the first and third axes, collapsing your ability to describe
certain orientations.

Think of a quaternion as "there is ONE axis in space, and I'm rotated
THIS much around it." There's always exactly one axis and one angle.
No sequential steps. No chance of two things aligning and cancelling out.

---

## Further reading (when you take classical mechanics)

- **Goldstein, Chapter 4**: The formal treatment of rigid-body kinematics,
  Euler angles, and the connection to rotation matrices.
- **Landau & Lifshitz, Mechanics, Chapter 6**: Angular momentum and
  Euler's equations in the body frame.
- The quaternion formulation we use bypasses Euler angles entirely.
  You'll learn Euler angles in class, then appreciate why nobody actually
  uses them in code.
