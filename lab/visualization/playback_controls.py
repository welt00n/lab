"""
PlaybackControls — generic animation control widgets.

Provides pause/resume, speed slider, and optional step forward/back
buttons. Reusable across any animated visualization.
"""

from __future__ import annotations

from matplotlib.widgets import Button, Slider


class PlaybackControls:
    """
    Pause / speed / step widgets for matplotlib FuncAnimation.

    Parameters
    ----------
    fig : matplotlib Figure
    on_pause : callable(paused: bool)
    on_speed : callable(speed: float)
    on_step_fwd, on_step_back : callable or None
        If provided, step buttons are added (replay mode).
    """

    def __init__(self, fig, on_pause, on_speed,
                 on_step_fwd=None, on_step_back=None):
        self.fig = fig
        self._paused = False

        has_step = on_step_fwd is not None

        if has_step:
            ax_back = fig.add_axes([0.65, 0.02, 0.06, 0.04])
            ax_play = fig.add_axes([0.72, 0.02, 0.06, 0.04])
            ax_pause = fig.add_axes([0.79, 0.02, 0.06, 0.04])
            ax_fwd = fig.add_axes([0.86, 0.02, 0.06, 0.04])

            self.btn_back = Button(ax_back, "<< Back")
            self.btn_play = Button(ax_play, "> Play")
            self.btn_pause = Button(ax_pause, "|| Pause")
            self.btn_fwd = Button(ax_fwd, ">> Step")

            self.btn_play.on_clicked(lambda _: on_pause(False))
            self.btn_pause.on_clicked(lambda _: on_pause(True))
            self.btn_fwd.on_clicked(lambda _: on_step_fwd())
            self.btn_back.on_clicked(lambda _: on_step_back())
        else:
            ax_pbtn = fig.add_axes([0.38, 0.015, 0.08, 0.035])
            ax_spd = fig.add_axes([0.50, 0.015, 0.18, 0.035])

            self.btn_pause = Button(ax_pbtn, "|| Pause")
            self.speed_slider = Slider(
                ax_spd, "speed", 0.25, 4.0,
                valinit=1.0, valstep=0.25, valfmt="x%.2g")

            def _toggle_pause(_):
                self._paused = not self._paused
                on_pause(self._paused)
                self.btn_pause.label.set_text(
                    "> Resume" if self._paused else "|| Pause")
                fig.canvas.draw_idle()

            self.btn_pause.on_clicked(_toggle_pause)
            self.speed_slider.on_changed(on_speed)
