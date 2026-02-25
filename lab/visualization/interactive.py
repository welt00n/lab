"""
Interactive parameter exploration with matplotlib sliders.
"""

import numpy as np


def explore_parameter(system_factory, param_name, param_range,
                      q0, p0, dt=0.01, duration=10.0,
                      plot_type="phase", coord=0):
    """
    Interactive slider for exploring how a parameter affects the dynamics.

    Parameters
    ----------
    system_factory : callable(**kwargs) -> Hamiltonian
        Factory function (e.g. ``oscillators.harmonic``) that accepts
        the parameter to vary.
    param_name : str
        Name of the parameter to vary.
    param_range : (float, float)
        Min and max values for the slider.
    q0, p0 : array-like
        Initial conditions.
    plot_type : "phase" | "trajectory" | "energy"
        What to display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from lab.core.experiment import Experiment

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.2, 0.08, 0.6, 0.04])
    init_val = (param_range[0] + param_range[1]) / 2
    slider = Slider(ax_slider, param_name, param_range[0], param_range[1],
                    valinit=init_val)

    def update(val):
        ax.clear()
        H = system_factory(**{param_name: val})
        exp = Experiment(H, q0=q0, p0=p0, dt=dt, duration=duration)
        data = exp.run()

        if plot_type == "phase":
            ax.plot(data.q[:, coord], data.p[:, coord])
            cname = H.coords[coord] if coord < len(H.coords) else f"q{coord}"
            ax.set_xlabel(cname)
            ax.set_ylabel(f"p_{cname}")
        elif plot_type == "trajectory":
            ax.plot(data.t, data.q[:, coord])
            ax.set_xlabel("time")
            ax.set_ylabel(H.coords[coord] if coord < len(H.coords) else f"q{coord}")
        elif plot_type == "energy":
            ax.plot(data.t, data.energy)
            ax.set_xlabel("time")
            ax.set_ylabel("energy")

        ax.set_title(f"{H.name}  ({param_name}={val:.4g})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(init_val)
    plt.show()
    return fig, slider


def explore_initial_conditions(hamiltonian, q_range, p_range,
                               dt=0.01, duration=10.0, coord=0):
    """
    Click on the phase plane to launch a trajectory from that (q, p).

    Parameters
    ----------
    q_range, p_range : (float, float)
        Axis limits.
    """
    import matplotlib.pyplot as plt
    from lab.core.experiment import Experiment

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*q_range)
    ax.set_ylim(*p_range)
    cname = (hamiltonian.coords[coord]
             if coord < len(hamiltonian.coords) else f"q{coord}")
    ax.set_xlabel(cname)
    ax.set_ylabel(f"p_{cname}")
    ax.set_title(f"{hamiltonian.name} — click to launch trajectories")

    def on_click(event):
        if event.inaxes != ax:
            return
        q0 = np.zeros(hamiltonian.ndof)
        p0 = np.zeros(hamiltonian.ndof)
        q0[coord] = event.xdata
        p0[coord] = event.ydata

        exp = Experiment(hamiltonian, q0=q0, p0=p0, dt=dt, duration=duration)
        data = exp.run()
        ax.plot(data.q[:, coord], data.p[:, coord], alpha=0.7, lw=0.8)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return fig
