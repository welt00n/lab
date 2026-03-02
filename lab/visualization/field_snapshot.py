"""
Field snapshot visualization — 2D colormesh, quiver plots, and
animations for FDTD electromagnetic wave simulations.
"""

import numpy as np


def plot_1d_snapshot(fdtd_data, frame=-1, ax=None, field="Ey"):
    """
    Plot a 1D field snapshot.

    Parameters
    ----------
    fdtd_data : FDTDDataSet
    frame : int
        Which snapshot to plot (-1 = last).
    field : "Ey" | "Bz"
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    data = getattr(fdtd_data, field)
    if data is None:
        raise ValueError(f"No {field} data in this dataset")

    ax.plot(fdtd_data.x, data[frame])
    ax.set_xlabel("x")
    ax.set_ylabel(field)
    ax.set_title(f"{field} at t = {fdtd_data.t[frame]:.4g}")
    ax.axhline(0, color="k", lw=0.3)
    return ax


def plot_2d_snapshot(fdtd_data, frame=-1, ax=None, cmap="RdBu_r",
                     vmax=None):
    """
    Plot a 2D Ez field snapshot as a colormesh.

    Parameters
    ----------
    fdtd_data : FDTDDataSet (ndim=2)
    frame : int
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    data = fdtd_data.Ez[frame]
    if vmax is None:
        vmax = np.max(np.abs(data)) or 1.0

    extent = [fdtd_data.x[0], fdtd_data.x[-1],
              fdtd_data.y[0], fdtd_data.y[-1]]

    im = ax.imshow(data.T, origin="lower", extent=extent,
                   cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Ez at t = {fdtd_data.t[frame]:.4g}")
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax, label="Ez")
    return ax


def animate_1d(fdtd_data, field="Ey", interval=30, save_path=None):
    """Animate a 1D FDTD simulation."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    data = getattr(fdtd_data, field)
    fig, ax = plt.subplots()
    ymax = np.max(np.abs(data)) * 1.1 or 1.0
    ax.set_xlim(fdtd_data.x[0], fdtd_data.x[-1])
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel(field)
    line, = ax.plot([], [])
    title = ax.set_title("")

    def update(frame):
        line.set_data(fdtd_data.x, data[frame])
        title.set_text(f"{field}  t = {fdtd_data.t[frame]:.4g}")
        return line, title

    anim = animation.FuncAnimation(fig, update, frames=len(fdtd_data.t),
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def animate_2d(fdtd_data, cmap="RdBu_r", interval=50, save_path=None):
    """Animate a 2D FDTD simulation."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    vmax = np.max(np.abs(fdtd_data.Ez)) or 1.0
    extent = [fdtd_data.x[0], fdtd_data.x[-1],
              fdtd_data.y[0], fdtd_data.y[-1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(fdtd_data.Ez[0].T, origin="lower", extent=extent,
                   cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Ez")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    def update(frame):
        im.set_data(fdtd_data.Ez[frame].T)
        title.set_text(f"Ez  t = {fdtd_data.t[frame]:.4g}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(fdtd_data.t),
                                   interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def plot_ray_paths(ray_datasets, n_func=None, ax=None, xlim=None, ylim=None):
    """
    Plot 2D ray paths from multiple DataSets (from ray optics experiments).

    Optionally overlay the refractive index field as a background.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if n_func is not None and xlim is not None and ylim is not None:
        xg = np.linspace(*xlim, 300)
        yg = np.linspace(*ylim, 300)
        X, Y = np.meshgrid(xg, yg)
        N = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                N[i, j] = n_func(np.array([X[i, j], Y[i, j]]))
        ax.pcolormesh(X, Y, N, cmap="Greys", alpha=0.3, shading="auto")

    for ds in ray_datasets:
        ax.plot(ds.q[:, 0], ds.q[:, 1], lw=0.8, alpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("ray paths")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    return ax
