"""
FDTD (Finite-Difference Time-Domain) Maxwell solver.

Solves Maxwell's equations on a discrete grid using the Yee lattice
algorithm.  E and B fields are staggered in both space and time, making
the scheme a natural leapfrog — the same symplectic philosophy as the
particle integrator.

Supports:
    - 1D and 2D grids
    - Dielectric and conducting materials via permittivity/conductivity maps
    - Sources: point, plane wave, oscillating dipole
    - PML (Perfectly Matched Layer) absorbing boundary conditions
    - Field snapshots returned as a DataSet-like object
"""

import numpy as np

from lab.core.dataset import DataSet
from lab.core.hamiltonian import Hamiltonian


class FDTDGrid1D:
    """
    1D FDTD simulation along the x-axis.

    Fields:
        Ey(x, t) — electric field (y-polarized)
        Bz(x, t) — magnetic field (z-component)

    Update equations (Gaussian units with c = 1 by default):
        Ey^{n+1} = Ey^n + (dt/dx)(Bz^{n+1/2}[i-1] - Bz^{n+1/2}[i]) / eps[i]
                   - sigma[i]*dt*Ey^n / eps[i] + J_source
        Bz^{n+1/2} = Bz^{n-1/2} + (dt/dx)(Ey^n[i] - Ey^n[i+1])
    """

    def __init__(self, nx=400, dx=0.01, dt=None, c=1.0):
        self.nx = nx
        self.dx = dx
        self.c = c
        self.dt = dt if dt is not None else 0.5 * dx / c  # CFL condition

        self.Ey = np.zeros(nx)
        self.Bz = np.zeros(nx)

        self.epsilon = np.ones(nx)
        self.sigma = np.zeros(nx)

        self.sources = []
        self.pml_width = 0
        self._pml_Ey = None
        self._pml_Bz = None
        self.time = 0.0
        self.step_count = 0

    def set_material(self, i_start, i_end, epsilon=1.0, sigma=0.0):
        """Set material properties in a region."""
        self.epsilon[i_start:i_end] = epsilon
        self.sigma[i_start:i_end] = sigma

    def add_source(self, source):
        """Add a source (callable(grid, step) -> modifies Ey in-place)."""
        self.sources.append(source)

    def enable_pml(self, width=20, sigma_max=None):
        """Enable PML absorbing boundaries."""
        self.pml_width = width
        if sigma_max is None:
            sigma_max = 0.8 * (3 + 1) / (self.dx * np.sqrt(1.0))

        profile = sigma_max * (np.arange(width) / width) ** 3
        self._pml_Ey = np.zeros(self.nx)
        self._pml_Bz = np.zeros(self.nx)

        self._pml_Ey[:width] = profile[::-1]
        self._pml_Ey[-width:] = profile
        self._pml_Bz[:width] = profile[::-1]
        self._pml_Bz[-width:] = profile

    def step(self):
        """Advance one time step."""
        dt, dx = self.dt, self.dx

        self.Bz[:-1] += (dt / dx) * (self.Ey[:-1] - self.Ey[1:])
        if self._pml_Bz is not None:
            self.Bz *= np.exp(-self._pml_Bz * dt)

        ca = (1.0 - self.sigma * dt / (2 * self.epsilon)) / \
             (1.0 + self.sigma * dt / (2 * self.epsilon))
        cb = (dt / (dx * self.epsilon)) / \
             (1.0 + self.sigma * dt / (2 * self.epsilon))

        self.Ey[1:] = ca[1:] * self.Ey[1:] + cb[1:] * (self.Bz[:-1] - self.Bz[1:])
        if self._pml_Ey is not None:
            self.Ey *= np.exp(-self._pml_Ey * dt)

        for src in self.sources:
            src(self, self.step_count)

        self.time += dt
        self.step_count += 1

    def run(self, nsteps, snapshot_interval=10, progress=False):
        """
        Run the simulation, returning field snapshots.

        Returns an FDTDDataSet with snapshots of Ey and Bz.
        """
        t_list = []
        Ey_list = []
        Bz_list = []

        for i in range(nsteps):
            if i % snapshot_interval == 0:
                t_list.append(self.time)
                Ey_list.append(self.Ey.copy())
                Bz_list.append(self.Bz.copy())
            self.step()
            if progress and i % max(1, nsteps // 20) == 0:
                print(f"  {100*i/nsteps:.0f}%")

        return FDTDDataSet(
            t=np.array(t_list),
            Ey=np.array(Ey_list),
            Bz=np.array(Bz_list),
            x=np.arange(self.nx) * self.dx,
            ndim=1,
        )


class FDTDGrid2D:
    """
    2D FDTD simulation in the xy-plane (TM mode).

    Fields:
        Ez(x, y, t) — electric field (z-polarized, out of plane)
        Hx(x, y, t) — magnetic field x-component
        Hy(x, y, t) — magnetic field y-component

    Standard Yee lattice staggering.
    """

    def __init__(self, nx=200, ny=200, dx=0.01, dy=None, dt=None, c=1.0):
        self.nx, self.ny = nx, ny
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.c = c
        self.dt = dt if dt is not None else 0.5 / (c * np.sqrt(1/dx**2 + 1/self.dy**2))

        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

        self.epsilon = np.ones((nx, ny))
        self.sigma = np.zeros((nx, ny))

        self.sources = []
        self.time = 0.0
        self.step_count = 0

    def set_material(self, region, epsilon=1.0, sigma=0.0):
        """
        Set material in a region.

        region: (x_start, x_end, y_start, y_end) index ranges.
        """
        xs, xe, ys, ye = region
        self.epsilon[xs:xe, ys:ye] = epsilon
        self.sigma[xs:xe, ys:ye] = sigma

    def add_source(self, source):
        self.sources.append(source)

    def step(self):
        dt, dx, dy = self.dt, self.dx, self.dy

        self.Hx[:, :-1] -= (dt / dy) * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:-1, :] += (dt / dx) * (self.Ez[1:, :] - self.Ez[:-1, :])

        ca = (1.0 - self.sigma * dt / (2*self.epsilon)) / \
             (1.0 + self.sigma * dt / (2*self.epsilon))
        cb_x = (dt / (dx * self.epsilon)) / \
               (1.0 + self.sigma * dt / (2*self.epsilon))
        cb_y = (dt / (dy * self.epsilon)) / \
               (1.0 + self.sigma * dt / (2*self.epsilon))

        self.Ez[1:, 1:] = (
            ca[1:, 1:] * self.Ez[1:, 1:]
            + cb_x[1:, 1:] * (self.Hy[1:, 1:] - self.Hy[:-1, 1:])
            - cb_y[1:, 1:] * (self.Hx[1:, 1:] - self.Hx[1:, :-1])
        )

        for src in self.sources:
            src(self, self.step_count)

        self.time += dt
        self.step_count += 1

    def run(self, nsteps, snapshot_interval=20, progress=False):
        t_list, Ez_list = [], []

        for i in range(nsteps):
            if i % snapshot_interval == 0:
                t_list.append(self.time)
                Ez_list.append(self.Ez.copy())
            self.step()
            if progress and i % max(1, nsteps // 20) == 0:
                print(f"  {100*i/nsteps:.0f}%")

        return FDTDDataSet(
            t=np.array(t_list),
            Ez=np.array(Ez_list),
            x=np.arange(self.nx) * self.dx,
            y=np.arange(self.ny) * self.dy,
            ndim=2,
        )


class FDTDDataSet:
    """Container for FDTD simulation output."""

    def __init__(self, t, x, ndim, y=None, Ey=None, Bz=None, Ez=None):
        self.t = t
        self.x = x
        self.y = y
        self.ndim = ndim
        self.Ey = Ey
        self.Bz = Bz
        self.Ez = Ez
        self.nsteps = len(t)

    def __repr__(self):
        return f"FDTDDataSet(ndim={self.ndim}, snapshots={self.nsteps})"


# ===================================================================
# Source factories
# ===================================================================

def gaussian_pulse(position_index, amplitude=1.0, width=30.0, delay=60):
    """1D Gaussian pulse source for Ey."""
    def source(grid, step):
        grid.Ey[position_index] += amplitude * np.exp(
            -((step - delay)**2) / (2 * width**2))
    return source


def sinusoidal_source(position_index, frequency, amplitude=1.0, c=1.0):
    """1D sinusoidal (CW) source for Ey."""
    omega = 2 * np.pi * frequency
    def source(grid, step):
        grid.Ey[position_index] += amplitude * np.sin(omega * step * grid.dt)
    return source


def point_source_2d(ix, iy, frequency, amplitude=1.0):
    """2D point source for Ez."""
    omega = 2 * np.pi * frequency
    def source(grid, step):
        grid.Ez[ix, iy] += amplitude * np.sin(omega * step * grid.dt)
    return source


def plane_wave_source_2d(y_index, frequency, amplitude=1.0):
    """2D plane wave source (uniform along x) for Ez."""
    omega = 2 * np.pi * frequency
    def source(grid, step):
        grid.Ez[:, y_index] += amplitude * np.sin(omega * step * grid.dt)
    return source
