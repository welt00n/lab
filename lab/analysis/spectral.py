"""
Spectral analysis — FFT of trajectories and power spectra.
"""

import numpy as np


def power_spectrum(dataset, coord=0, ax=None):
    """
    Compute and plot the power spectrum of q_i(t).

    Returns (frequencies, power).
    """
    import matplotlib.pyplot as plt

    dt = dataset.t[1] - dataset.t[0] if len(dataset.t) > 1 else 1.0
    signal = dataset.q[:, coord]
    signal = signal - np.mean(signal)

    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)**2 / N
    freqs = np.fft.rfftfreq(N, d=dt)

    if ax is not None or True:
        if ax is None:
            _, ax = plt.subplots()
        ax.semilogy(freqs[1:], power[1:])
        cname = (dataset.hamiltonian.coords[coord]
                 if coord < len(dataset.hamiltonian.coords) else f"q{coord}")
        ax.set_xlabel("frequency")
        ax.set_ylabel("power")
        ax.set_title(f"power spectrum — {cname}")

    return freqs, power, ax


def dominant_frequency(dataset, coord=0):
    """Return the frequency with the highest power."""
    freqs, power, _ = power_spectrum(dataset, coord=coord)
    idx = np.argmax(power[1:]) + 1
    return float(freqs[idx])


def spectrogram(dataset, coord=0, window_size=256, overlap=128, ax=None):
    """
    Short-time Fourier transform spectrogram.

    Useful for systems with time-varying frequency content (driven
    oscillators, chirps).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    dt = dataset.t[1] - dataset.t[0]
    signal = dataset.q[:, coord] - np.mean(dataset.q[:, coord])
    step = window_size - overlap
    n_windows = max(1, (len(signal) - window_size) // step + 1)

    spectra = []
    times = []
    for i in range(n_windows):
        start = i * step
        chunk = signal[start:start + window_size]
        if len(chunk) < window_size:
            break
        window = np.hanning(window_size) * chunk
        fft_vals = np.fft.rfft(window)
        spectra.append(np.abs(fft_vals)**2)
        times.append(dataset.t[start + window_size // 2])

    if not spectra:
        return ax

    spectra = np.array(spectra).T
    freqs = np.fft.rfftfreq(window_size, d=dt)

    ax.pcolormesh(times, freqs, np.log10(spectra + 1e-20),
                  shading="auto", cmap="inferno")
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")
    ax.set_title(f"spectrogram — {dataset.hamiltonian.coords[coord]}")
    return ax
