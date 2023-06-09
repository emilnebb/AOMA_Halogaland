import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import koma.modal as modal
from dataloader import FEM_result_loader
import colordict
color_values = list(colordict.ColorDict(norm=1).values())

def welch_plot(acc, sampling_frequency, Ndivisions):
    """
    Creates a Power Spectral Density plot of the input data estimated by Welch's method.
    :param acc: acceleration data, 1d np.array.
    :param sampling_frequency: int
    :param Ndivisions:int
    :return:
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
    plt.ylabel('$S(f)$')  #
    # plt.xlim([0,5])
    # plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

def stabilization_diagram(acceleration, sampling_frequency, Ndivisions, frequencies, orders,
                          all_freqs = None, all_orders = None):
    """
    Creates a stabilization diagram with clustered poles labeled with the same color.
    Optional takes the discarded poles and plots them as grey dots in the stabilization diagram.
    Frequency content of the acceleration data is decomposed into horizontal, vertical and tortional
    directions.
    :param acceleration: 2d np.array of acceleration data
    :param sampling_frequency: int
    :param Ndivisions: int
    :param frequencies: frequencies of the clustered poles, 2d np.array
    :param orders: orders of the clustered poles, 2d np.array
    :param all_freqs: frequecies of all poles created from cov-SSI, 1d np.array
    :param all_orders: orders of all poles creaed from cov-SSI, 1d np.array
    :return: figure including the stabilization diagram
    """

    fig, ax =plt.subplots(figsize=(14, 5), dpi=300)

    if len(all_freqs) > 0 and len(all_orders) > 0:
        ax.scatter(all_freqs, all_orders, marker= 'o', color='grey', label='Discarded modes')

    for cluster in range(frequencies.shape[0]):
        ax.plot(frequencies[cluster], orders[cluster], marker='o', color=color_values[cluster +cluster*3]) #,
                #label=str(np.round(np.mean(frequencies[cluster]), decimals=3)) + " Hz")

    ax.set_ylabel("Model order")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_xticks(np.arange(0, 1.1, step=0.1))
    ax.set_xlim([0, 1])

    #Creating power spectral density in each direction of motion of the bridge
    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    length = acceleration.shape[1]
    length = int(length/3)
    n_channels = int(length/2)

    B = 18.6
    # Transforming to find y, z, theta component for the sensor pair at the midspan
    y = (acceleration[:, length + int(np.floor(length/4))] +
         acceleration[:, length + int(np.floor(length/4)) + n_channels]) / 2
    z = (acceleration[:, length + int(np.floor(length/4)) + n_channels*2] +
         acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / 2
    theta = (-acceleration[:, length + int(np.floor(length/4)) + n_channels*2] +
             acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / B

    # Call welch from scipy signal processing
    f, Sy_welch = signal.welch(y, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Sz_welch = signal.welch(z, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Stheta_welch = signal.welch(theta, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')

    #Sx_est = np.abs((2.0 / len(acceleration)) * np.fft.rfft(acceleration))**2*1000
    #f = np.fft.rfftfreq(acceleration.shape[0], dt)


    ax2 = ax.twinx()
    ax2.plot(f, Sy_welch, color='black', label='$PSD\ y-direction$', lw=0.5)
    ax2.plot(f, Sz_welch, color='blue', label='$PSD\ z-direction$', lw=0.5)
    ax2.plot(f, Stheta_welch*10, color='red', label=r'$PSD\ \theta-direction\ e1$', lw=0.5)
    ax2.set_ylabel("PSD")
    #ax2.set_yscale('symlog')
    #ax2.legend()

    plt.legend()
    plt.grid()

    return fig

def stabilization_diagram_cov_ssi(acceleration, sampling_frequency, Ndivisions,
                          all_freqs, all_orders, freq_stab=None, orders_stab=None):
    """
    Creates a stabilization diagram with clustered poles labeled with the same color.
    Optional takes the discarded poles and plots them as grey dots in the stabilization diagram.
    Frequency content of the acceleration data is decomposed into horizontal, vertical and tortional
    directions.
    :param acceleration: 2d np.array of acceleration data
    :param sampling_frequency: int
    :param Ndivisions: int
    :param frequencies: frequencies of the clustered poles, 2d np.array
    :param orders: orders of the clustered poles, 2d np.array
    :param all_freqs: frequecies of all poles created from cov-SSI, 1d np.array
    :param all_orders: orders of all poles creaed from cov-SSI, 1d np.array
    :return: figure including the stabilization diagram
    """

    fig, ax =plt.subplots(figsize=(14, 5), dpi=300)
    ax.scatter(all_freqs, all_orders, marker= 'o', color='grey')

    if not (freq_stab is None) and not (orders_stab is None):
        ax.scatter(freq_stab, orders_stab, marker='o', color='blue')

    ax.set_ylabel("Model order")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_xticks(np.arange(0, 1, step=0.1))
    ax.set_xlim([0,1])

    #Creating power spectral density in each direction of motion of the bridge
    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    length = acceleration.shape[1]
    length = int(length/3)
    n_channels = int(length/2)

    B = 18.6
    # Transforming to find y, z, theta component for the sensor pair at the midspan
    y = (acceleration[:, length + int(np.floor(length/4))] + acceleration[:, length + int(np.floor(length/4)) +
                                                                             n_channels]) / 2
    z = (acceleration[:, length + int(np.floor(length/4)) + n_channels*2] +
         acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / 2
    theta = (-acceleration[:, length + int(np.floor(length/4)) + n_channels*2] +
             acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / B

    # Call welch from scipy signal processing
    f, Sy_welch = signal.welch(y, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Sz_welch = signal.welch(z, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
    f, Stheta_welch = signal.welch(theta, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')



    ax2 = ax.twinx()
    ax2.plot(f, Sy_welch, color='black', label='$PSD\ y-direction$', lw=0.5)
    ax2.plot(f, Sz_welch, color='blue', label='$PSD\ z-direction$', lw=0.5)
    ax2.plot(f, Stheta_welch*10, color='red', label=r'$PSD\ \theta-direction\ e1$', lw=0.5)
    ax2.set_ylabel("PSD")
    #ax2.legend()

    plt.legend()
    plt.grid()

    return fig


def plotModeShape(phi, i_phi_plot):
    """
    Arguments:
    phi        : mode shape matrix, each column containing mode shape values for x1, x2, y1, y2, z1, z2
                 at the sensor locations
    i_phi_plot : index of which mode shape to plot
    Returns:
    fig        : plot of mode shapes
    ---------------------------------
    Function plots mode shapes obtained from Cov-SSI
    """
    B = 18.6  # Width of bridge girder

    phi_x = phi[:16]
    phi_y = phi[16:32, :]
    phi_z = phi[32:48, :]

    phi_x = modal.maxreal((phi_x[::2, :] + phi_x[1::2, :]) / 2)
    phi_y = modal.maxreal((phi_y[::2, :] + phi_y[1::2, :]) / 2)
    phi_z = modal.maxreal((phi_z[::2, :] + phi_z[1::2, :]) / 2)

    phi_theta = modal.maxreal((-phi[32:40, :] + phi[40:48, :]) / B)

    x_sensors = np.array([-572.5, -420, -300, -180, -100, 0, 100, 260, 420,
                          572.5])  # Sensor x-coordinates - [TOWER, A03, A04, A05, A06, A07, A08, A09, A10, TOWER]
    # Calculating common y-lim for all plots
    ylim = 1.09 * max([max(abs(phi_x[:, i_phi_plot])), max(abs(phi_y[:, i_phi_plot])), max(abs(phi_z[:, i_phi_plot])),
                       max(abs(phi_theta[:, i_phi_plot]))])
    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(7, 10))
    axs[0].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_x[:, i_phi_plot]), np.array([0]))), color='black')
    axs[0].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[0].axvline(0, color='grey', linestyle=':', linewidth=1)
    axs[0].set_title('Longitudinal mode shape')
    axs[0].set_ylim(-ylim, ylim)

    axs[1].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_y[:, i_phi_plot]), np.array([0]))), color='black')
    axs[1].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[1].axvline(0, color='grey', linestyle=':', linewidth=1)
    axs[1].set_title('Horizontal mode shape')
    axs[1].set_ylim(-ylim, ylim)

    axs[2].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_z[:, i_phi_plot]), np.array([0]))), color='black')
    axs[2].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[2].axvline(0, color='grey', linestyle=':', linewidth=1)
    axs[2].set_title('Vertical mode shape')
    axs[2].set_ylim(-ylim, ylim)

    axs[3].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_theta[:, i_phi_plot]), np.array([0]))),
                color='black')
    axs[3].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[3].axvline(0, color='grey', linestyle=':', linewidth=1)
    axs[3].set_title('Torsional mode shape')
    axs[3].set_ylim(-ylim, ylim)

    plt.tight_layout()

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
    fig, axs = plt.subplots(int(np.ceil(num/2)), 2, figsize=(20, int(np.ceil(num/2))*3), dpi=300)

    j = 0
    for i in range(len(f)):
        axs[int(np.floor(j / 2)), j % 2].set_xlabel('x[m]')
        axs[int(np.floor(j / 2)), j % 2].set_ylim([-1, 1])
        axs[int(np.floor(j / 2)), j % 2].set_xlim([-600, 600])
        axs[int(np.floor(j / 2)), j % 2].set_xticks([-600, -300, 0, 300, 600])
        axs[int(np.floor(j / 2)), j % 2].set_yticks([-1, -0.5, 0, 0.5, 1])


        if FEM_loader.mode_type[i] == 'Horizontal' and type == 'Horizontal':
            factor = 1 / np.max(np.abs(phi_y[:, i]))
            axs[int(np.floor(j/2)), j % 2].plot(x, phi_y[:, i]*factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1
        elif FEM_loader.mode_type[i] == 'Vertical' and type == 'Vertical':
            factor = 1 / np.max(np.abs(phi_z[:, i]))
            axs[int(np.floor(j/2)), j % 2].plot(x, phi_z[:, i]*factor, color='black')
            axs[int(np.floor(j / 2)), j % 2].set_title(
                'Mode ' + str(i + 1) + ' - ' + type + '\n $f_n$ = ' + f"{f[i]:.3f}" + ' Hz')
            axs[int(np.floor(j / 2)), j % 2].grid()
            j += 1
        elif FEM_loader.mode_type[i] == 'Torsional' and type == 'Torsional':
            factor = 1 / np.max(np.abs(phi_t[:, i]))
            axs[int(np.floor(j/2)), j % 2].plot(x, phi_t[:, i]*factor, color='black')
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
        fig.delaxes(axs[int(np.ceil(num/2))-1, 1])

    fig.tight_layout()

    return  fig

