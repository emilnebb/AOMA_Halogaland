import numpy as np
import strid
import matplotlib.pyplot as plt
import scipy.signal

def generate_data(num_stories: int, path: str):
    """
    Generating random vibration data for a shear fram of num_storeys storeys.
    Based on example notebook "00-generating-data.ipynb".
    Parameters
    ----------
    num_stories: int
    path: str of path to save the data
    Returns
    -------
    Saves data to file.
    """
    # Create a shear frame
    sf = strid.utils.ShearFrame(num_stories, 1e3, 1e4)  # Defining 8 storeys here
    sf.set_rayleigh_damping_matrix([sf.get_natural_frequency(1), sf.get_natural_frequency(sf.n)], [.05] * 2)

    # Determine the time discretization and period
    Tmax = 1. / strid.w2f(sf.get_natural_frequency(1))
    fmax = strid.w2f(sf.get_natural_frequency(sf.n))
    T = 1000 * Tmax
    fs = 5 * fmax
    t = np.arange(0., T, 1 / fs)

    # Define loads on system
    ## Unmeasureable: Stochastic loads on all floors (Process noise)
    w = np.random.normal(size=(sf.n, t.size)) * 1e-1

    ## Load matrix, f
    F = w.copy()

    # Simulate response, accelerations at each floor measured
    y0, _, _ = sf.simulate(t, F)

    noise_std = y0.std()

    # Add measurement noise
    v = np.random.normal(size=y0.shape) * noise_std
    y = y0 + v

    """
    plt.figure("Accelerations measured at top floor")
    plt.plot(t, y[-1], label="w/noise")
    plt.plot(t, y0[-1], label="wo/noise")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.figure("PSD of accelerations at top floor")
    for yi in [y[-1], y0[-1]]:
        freqs, Gyy = scipy.signal.welch(yi, fs, nperseg=2 ** 10)
        plt.semilogy(freqs, Gyy)

    for n in range(1, 1 + sf.n):
        plt.axvline(strid.w2f(sf.get_natural_frequency(n)), alpha=.3)
    plt.ylabel('PSD')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    """


    true_frequencies = np.array([sf.get_natural_frequency(i) / (2 * np.pi) for i in range(1, sf.n + 1)])
    true_damping = np.array([sf.get_rayleigh_damping_ratio(i) for i in range(1, sf.n + 1)])
    true_modeshapes = np.array([sf.get_mode_shape(i) for i in range(1, sf.n + 1)])

    # Saving the data
    np.savez(path,
             y=y, fs=fs,
             true_frequencies=true_frequencies,
             true_damping=true_damping,
             true_modeshapes=true_modeshapes
             )