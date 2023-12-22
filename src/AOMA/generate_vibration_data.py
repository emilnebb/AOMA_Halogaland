import numpy as np
import strid


def generate_data(num_stories: int, path: str):
    """
    Generate simulated data for a shear frame system and save it to a file.

    Args:
        num_stories (int): The number of stories in the shear frame system.
        path (str): The path where the generated data will be saved.

    Returns:
        None
    """

    # Create a shear frame
    sf = strid.utils.ShearFrame(num_stories, 1e3, 1e4)
    sf.set_rayleigh_damping_matrix([sf.get_natural_frequency(1), sf.get_natural_frequency(sf.n)], [.05] * 2)

    # Determine the time discretization and period
    Tmax = 1. / strid.w2f(sf.get_natural_frequency(1))
    fmax = strid.w2f(sf.get_natural_frequency(sf.n))
    T = 100 * Tmax
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