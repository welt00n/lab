"""
Coin toss experiment — drop a coin from height h with varying
initial orientations and classify the outcome as heads or tails.

The coin is a thin rigid body disk lying in the body xz-plane.
Its normal is the body y-axis.

    Heads:  body y-axis points up in world frame after settling
    Tails:  body y-axis points down in world frame after settling

The key physics: at low heights, the outcome is almost entirely
determined by the initial tilt angle (deterministic regime).  As
height increases, the coin has time to tumble and the boundary
between heads and tails becomes fractal-like — sensitivity to
initial conditions emerges.
"""

import numpy as np

from lab.core import quaternion as quat
from lab.systems.rigid_body.objects import RigidBody
from lab.systems.rigid_body.fields import GravityField
from lab.systems.rigid_body.constraints import FloorConstraint
from lab.systems.rigid_body.world import World


# ===================================================================
# Outcome classification
# ===================================================================

HEADS = 1
TAILS = -1
EDGE = 0


def classify_final_orientation(orientation):
    """
    Given a unit quaternion [w, x, y, z], determine if the coin
    landed heads or tails by checking which direction the body
    y-axis points in world frame.
    """
    body_normal = np.array([0.0, 1.0, 0.0])
    world_normal = quat.rotate_vector(orientation, body_normal)
    y_component = world_normal[1]

    if y_component > 0.1:
        return HEADS
    elif y_component < -0.1:
        return TAILS
    return EDGE


# ===================================================================
# Single toss
# ===================================================================

def toss_coin(height, tilt_axis="x", tilt_angle=0.0,
              radius=0.05, mass=0.01, restitution=0.5,
              friction=0.6, rolling_resistance=0.05,
              g=9.81, dt=0.0005, duration=None,
              angular_momentum=None,
              record_trajectory=False):
    """
    Drop a coin from *height* with an initial tilt of *tilt_angle*
    radians about *tilt_axis*.

    Parameters
    ----------
    height : float
        Drop height (metres).
    tilt_axis : "x" | "y" | "z"
        Which world axis to rotate the initial orientation around.
    tilt_angle : float
        Tilt angle in radians.  0 = flat (heads up), π = flat (tails up),
        π/2 = perfectly on edge.
    radius : float
        Coin radius.
    mass : float
        Coin mass.
    restitution : float
        Coefficient of restitution for floor bounces.
    friction : float
        Coulomb friction coefficient at the floor contact.
    rolling_resistance : float
        Fraction of angular momentum removed per contact event.
    g : float
        Gravitational acceleration.
    dt : float
        Integration timestep.
    duration : float or None
        Simulation duration.  If None, estimated from free-fall time
        with a generous safety margin.
    angular_momentum : ndarray or None
        Initial angular momentum in body frame.
    record_trajectory : bool
        If True, also return time-series data.

    Returns
    -------
    result : int
        HEADS (1), TAILS (-1), or EDGE (0).
    trajectory : dict (only if record_trajectory=True)
        Keys: "t", "y", "orientation", "energy", "vy".
    """
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    axis_vec = axis_map[tilt_axis]
    orientation = quat.from_axis_angle(axis_vec, tilt_angle)

    body = RigidBody.coin(
        mass=mass,
        radius=radius,
        position=np.array([0.0, height, 0.0]),
        momentum=np.zeros(3),
        orientation=orientation,
        angular_momentum=angular_momentum,
    )

    world = World()
    world.add_particle(body)
    world.add_field(GravityField(g))
    world.add_constraint(FloorConstraint(
        restitution, friction=friction,
        rolling_resistance=rolling_resistance))

    if duration is None:
        t_fall = np.sqrt(2 * height / g)
        duration = max(t_fall * 8, 2.0)

    nsteps = int(np.ceil(duration / dt))

    traj = None
    if record_trajectory:
        traj = {
            "t": [0.0],
            "y": [body.position[1]],
            "orientation": [body.orientation.copy()],
            "energy": [world.total_energy()],
            "vy": [body.velocity[1]],
        }

    settled_count = 0
    settle_threshold = 1e-6

    for i in range(nsteps):
        world.step(dt)

        if record_trajectory:
            traj["t"].append(world.time)
            traj["y"].append(body.position[1])
            traj["orientation"].append(body.orientation.copy())
            traj["energy"].append(world.total_energy())
            traj["vy"].append(body.velocity[1])

        ke = body.kinetic_energy()
        if ke < settle_threshold and body.position[1] < radius + 0.05:
            settled_count += 1
            if settled_count > 200:
                break
        else:
            settled_count = 0

    result = classify_final_orientation(body.orientation)

    if record_trajectory:
        for k in traj:
            traj[k] = np.array(traj[k])
        return result, traj

    return result


# ===================================================================
# Parameter sweep
# ===================================================================

def sweep_h_vs_angle(heights, angles, tilt_axis="x", **kwargs):
    """
    Run a 2D grid of coin tosses: height × tilt angle.

    Parameters
    ----------
    heights : array-like
        Heights to sweep.
    angles : array-like
        Tilt angles to sweep (radians).
    tilt_axis : str
        "x", "y", or "z".
    **kwargs
        Passed to toss_coin().

    Returns
    -------
    results : ndarray, shape (len(heights), len(angles))
        HEADS (1), TAILS (-1), or EDGE (0) for each (h, angle) pair.
    """
    heights = np.asarray(heights)
    angles = np.asarray(angles)
    results = np.zeros((len(heights), len(angles)), dtype=int)

    total = len(heights) * len(angles)
    done = 0

    for i, h in enumerate(heights):
        for j, angle in enumerate(angles):
            results[i, j] = toss_coin(
                height=h, tilt_axis=tilt_axis, tilt_angle=angle,
                **kwargs,
            )
            done += 1
            if done % max(1, total // 10) == 0:
                print(f"  {100 * done / total:.0f}%  ({done}/{total})")

    return results


def plot_outcome_map(heights, angles, results, tilt_axis="x", ax=None):
    """
    Plot the heads/tails outcome map as a colour image.

    Heights on y-axis, tilt angles on x-axis.
    Blue = heads, red = tails, white = edge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    cmap = ListedColormap(["#d62728", "white", "#1f77b4"])
    extent = [
        np.degrees(angles[0]), np.degrees(angles[-1]),
        heights[0], heights[-1],
    ]

    ax.imshow(
        results, origin="lower", aspect="auto",
        extent=extent, cmap=cmap, vmin=-1, vmax=1,
        interpolation="nearest",
    )

    ax.set_xlabel(f"initial tilt about {tilt_axis}-axis (degrees)")
    ax.set_ylabel("drop height (m)")
    ax.set_title("Coin toss outcome map")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Heads"),
        Patch(facecolor="#d62728", label="Tails"),
        Patch(facecolor="white", edgecolor="gray", label="Edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    return fig, ax


def plot_single_toss(traj, result, ax=None):
    """Plot height vs time for a single coin toss."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    label = {HEADS: "Heads", TAILS: "Tails", EDGE: "Edge"}[result]
    color = {HEADS: "#1f77b4", TAILS: "#d62728", EDGE: "gray"}[result]

    ax.plot(traj["t"], traj["y"], color=color, lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("height (m)")
    ax.set_title(f"Coin drop — result: {label}")
    ax.grid(True, alpha=0.3)

    return ax
