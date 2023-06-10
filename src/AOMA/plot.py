import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import koma.modal as modal
from dataloader import FEM_result_loader
import colordict

color_values = list(colordict.ColorDict(norm=1).values())


def welch_plot(acc, sampling_frequency, Ndivisions):
    """
    Compute and plot the Welch spectrum of the acceleration data.

    Args:
        acc (array-like): The acceleration data.
        sampling_frequency (float): The sampling frequency of the acceleration data.
        Ndivisions (int): The number of divisions/segments for computing the Welch spectrum.

    Returns:
        None
    """

    Nwindow = np.ceil(len(acc) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    # Call welch from scipy signal processing
    f, Sx_welch = signal.welch(acc, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')

    plt.figure(figsize=(14, 7), dpi=250)
    plt.plot(f, Sx_welch, label='Welch spectrum of acceleration data')
    plt.xlabel('$f$ [Hz]')
    plt.ylabel('$S(f)$')
    plt.grid()
    plt.legend()
    plt.show()


def stabilization_diagram(acceleration, sampling_frequency, Ndivisions, frequencies, orders,
                          all_freqs=None, all_orders=None):
    """
    Plot the stabilization diagram showing model orders and frequencies, along with power spectral density (PSD)
    of the acceleration data.

    Args:
        acceleration (array-like): The acceleration data.
        sampling_frequency (float): The sampling frequency of the acceleration data.
        Ndivisions (int): The number of divisions/segments for computing the PSD.
        frequencies (array-like): An array of frequencies for each cluster.
        orders (array-like): An array of model orders for each cluster.
        all_freqs (array-like, optional): An array of frequencies for discarded modes (default: None).
        all_orders (array-like, optional): An array of model orders for discarded modes (default: None).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)

    if len(all_freqs) > 0 and len(all_orders) > 0:
        ax.scatter(all_freqs, all_orders, marker='o', color='grey', label='Discarded modes')

    for cluster in range(frequencies.shape[0]):
        ax.plot(frequencies[cluster], orders[cluster], marker='o', color=color_values[cluster + cluster * 3])

    ax.set_ylabel("Model order")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_xticks(np.arange(0, 1.1, step=0.1))
    ax.set_xlim([0, 1])

    # Creating power spectral density in each direction of motion of the bridge
    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    length = acceleration.shape[1]
    length = int(length / 3)
    n_channels = int(length / 2)

    B = 18.6
    # Transforming to find y, z, theta component for the sensor pair at the midspan
    y = (acceleration[:, length + int(np.floor(length / 4))] +
         acceleration[:, length + int(np.floor(length / 4)) + n_channels]) / 2
    z = (acceleration[:, length + int(np.floor(length / 4)) + n_channels * 2] +
         acceleration[:, length + int(np.floor(length / 4)) + n_channels * 3]) / 2
    theta = (-acceleration[:, length + int(np.floor(length / 4)) + n_channels * 2] +
             acceleration[:, length + int(np.floor(length / 4)) + n_channels * 3]) / B

    # Call welch from scipy signal processing
    f, Sy_welch = signal.welch(y, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Sz_welch = signal.welch(z, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Stheta_welch = signal.welch(theta, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                                   detrend='constant', return_onesided=True, scaling='density', axis=- 1,
                                   average='mean')

    ax2 = ax.twinx()
    ax2.plot(f, Sy_welch, color='black', label='$PSD\ y-direction$', lw=0.5)
    ax2.plot(f, Sz_welch, color='blue', label='$PSD\ z-direction$', lw=0.5)
    ax2.plot(f, Stheta_welch * 10, color='red', label=r'$PSD\ \theta-direction\ e1$', lw=0.5)
    ax2.set_ylabel("PSD")
    plt.legend()
    plt.grid()

    return fig


def stabilization_diagram_cov_ssi(acceleration, sampling_frequency, Ndivisions,
                                  all_freqs, all_orders, freq_stab=None, orders_stab=None):
    """
    Plot the stabilization diagram for covariance-driven subspace identification (SSI) method,
    showing model orders and frequencies, along with power spectral density (PSD)
    of the acceleration data.

    Args:
        acceleration (array-like): The acceleration data.
        sampling_frequency (float): The sampling frequency of the acceleration data.
        Ndivisions (int): The number of divisions/segments for computing the PSD.
        all_freqs (array-like): An array of frequencies for all the identified modes.
        all_orders (array-like): An array of model orders for all the identified modes.
        freq_stab (array-like, optional): An array of frequencies for stable modes (default: None).
        orders_stab (array-like, optional): An array of model orders for stable modes (default: None).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.scatter(all_freqs, all_orders, marker='o', color='grey')

    if not (freq_stab is None) and not (orders_stab is None):
        ax.scatter(freq_stab, orders_stab, marker='o', color='blue')

    ax.set_ylabel("Model order")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_xticks(np.arange(0, 1, step=0.1))
    ax.set_xlim([0, 1])

    # Creating power spectral density in each direction of motion of the bridge
    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    length = acceleration.shape[1]
    length = int(length / 3)
    n_channels = int(length / 2)

    B = 18.6
    # Transforming to find y, z, theta component for the sensor pair at the midspan
    y = (acceleration[:, length + int(np.floor(length / 4))] + acceleration[:, length + int(np.floor(length / 4)) +
                                                                               n_channels]) / 2
    z = (acceleration[:, length + int(np.floor(length / 4)) + n_channels * 2] +
         acceleration[:, length + int(np.floor(length / 4)) + n_channels * 3]) / 2
    theta = (-acceleration[:, length + int(np.floor(length / 4)) + n_channels * 2] +
             acceleration[:, length + int(np.floor(length / 4)) + n_channels * 3]) / B

    # Call welch from scipy signal processing
    f, Sy_welch = signal.welch(y, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Sz_welch = signal.welch(z, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Stheta_welch = signal.welch(theta, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                                   detrend='constant', return_onesided=True, scaling='density', axis=- 1,
                                   average='mean')

    ax2 = ax.twinx()
    ax2.plot(f, Sy_welch, color='black', label='$PSD\ y-direction$', lw=0.5)
    ax2.plot(f, Sz_welch, color='blue', label='$PSD\ z-direction$', lw=0.5)
    ax2.plot(f, Stheta_welch * 10, color='red', label=r'$PSD\ \theta-direction\ e1$', lw=0.5)
    ax2.set_ylabel("PSD")
    plt.legend()
    plt.grid()

    return fig


def plotModeShapeFEM(FEM_loader: FEM_result_loader, type='Vertical'):
    """
    Plots the mode shapes of the Finite Element Model (FEM) loaded using FEM_result_loader.

    Args:
        FEM_loader (FEM_result_loader): An instance of the FEM_result_loader class.
        type (str, optional): The type of mode shape to plot. Defaults to 'Vertical'.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the mode shape plots.
    """

    f = FEM_loader.f
    phi_y = FEM_loader.phi_y
    phi_z = FEM_loader.phi_z
    phi_t = FEM_loader.phi_t
    x = FEM_loader.x_plot

    num = FEM_loader.mode_type.count(type)

    # Plot
    fig, axs = plt.subplots(int(np.ceil(num / 2)), 2, figsize=(20, int(np.ceil(num / 2)) * 3), dpi=300)

    j = 0
    for i in range(len(f)):
        axs[int(np.floor(j / 2)), j % 2].set_xlabel('x[m]')
        axs[int(np.floor(j / 2)), j % 2].set_ylim([-1, 1])
        axs[int(np.floor(j / 2)), j % 2].set_xlim([-600, 600])
        axs[int(np.floor(j / 2)), j % 2].set_xticks([-600, -300, 0, 300, 600])
        axs[int(np.floor(j / 2)), j % 2].set_yticks([-1, -0.5, 0, 0.5, 1])

        if FEM_loader.mode_type[i] == 'Horizontal' and type == 'Horizontal':
            factor = 1 / np.max(np.abs(phi_y[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x, phi_y[:, i] * factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1
        elif FEM_loader.mode_type[i] == 'Vertical' and type == 'Vertical':
            factor = 1 / np.max(np.abs(phi_z[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x, phi_z[:, i] * factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1
        elif FEM_loader.mode_type[i] == 'Torsional' and type == 'Torsional':
            factor = 1 / np.max(np.abs(phi_t[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x, phi_t[:, i] * factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1
        elif FEM_loader.mode_type[i] == 'Cable' and type == 'Cable':
            factor = 1 / np.max(np.abs(phi_y[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x, phi_y[:, i] * factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1

    if num % 2:
        fig.delaxes(axs[int(np.ceil(num / 2)) - 1, 1])

    fig.tight_layout()

    return fig
