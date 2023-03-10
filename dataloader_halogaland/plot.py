import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
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
    #ax.legend()

    #Creating power spectral density in each direction of motion of the bridge
    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    length = acceleration.shape[1]
    length = int(length/3)
    n_channels = int(length/2)

    B = 18.6
    # Transforming to find y, z, theta component for the sensor pair at the midspan
    y = (acceleration[:, length + int(np.floor(length/4))] + acceleration[:, length + int(np.floor(length/4)) + n_channels]) / 2
    z = (acceleration[:, length + int(np.floor(length/4)) + n_channels*2] + acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / 2
    theta = (-acceleration[:, length + int(np.floor(length/4)) + n_channels*2] + acceleration[:, length + int(np.floor(length/4)) + n_channels*3]) / B

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
    #ax2.legend()

    plt.legend()
    plt.grid()

    return fig

class BridgeModelHalogaland:

    def __init__(self):
        B = 18.6
        sensor_locations_x = np.array([-580, -420, -300, -180, -100, 0, 100, 260, 420, 580])
        x = np.vstack((sensor_locations_x, sensor_locations_x))
        y = np.vstack((np.zeros_like(sensor_locations_x), np.ones_like(sensor_locations_x)*B))
        z = np.ones_like(y)*30
        tower1_cordinates = np.array([[[-580, -580, -580], [-580, -580, -580]],
                                      [[0, 0, B/2], [B, B, B/2]],
                                      [[0, 30, 170], [0, 30, 170]]])
        tower2_cordinates = np.array([[[580, 580, 580], [580, 580, 580]],
                                      [[0, 0, B / 2], [B, B, B / 2]],
                                      [[0, 30, 170], [0, 30, 170]]])

        base_figure = plt.figure(figsize=(14, 5), dpi=300)
        ax = base_figure.add_subplot(projection='3d')
        ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
        ax.plot_wireframe(tower1_cordinates[0,:,:], tower1_cordinates[1,:,:], tower1_cordinates[2,:,:])
        ax.plot_wireframe(tower2_cordinates[0, :, :], tower2_cordinates[1, :, :], tower2_cordinates[2, :, :])
        plt.show()

    def show_underformed_geometry(self):
        plt.show()