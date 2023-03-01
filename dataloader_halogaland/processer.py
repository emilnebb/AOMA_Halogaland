import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import colordict
color_values = list(colordict.ColorDict(norm=1).values())

def low_pass(old_signal: np.ndarray, sampling_frequency, cutoff_frequency, filter_order) -> np.ndarray:
    """
    Apply Butterworth filter as lowpass filter
    :param signal: np.ndarray of signal
    :param sampling_frequency: int sampling frequency of the signal
    :param cutoff_frequency: int cutoff frequency in Hz, which frequency of higher content will be removed
    :param filter_order: int hyperparameter to the Butterworth filter
    :return: np.ndarray filtered signal
    """

    sos = signal.butter(filter_order, cutoff_frequency, btype='lowpass',
                        fs=sampling_frequency, output='sos')

    filtered_signal = signal.sosfilt(sos, old_signal)

    return filtered_signal

def downsample(sampling_frequency_old, old_signal: np.ndarray, sampling_frequency_new) -> np.ndarray:
    """
    Downsampling of main signal with corresponding time vector.
    :param timestamp: np.ndarray of timestamps of the main signal
    :param old_signal: np.ndarray of main signal
    :param sampling_frequency_new: int
    :return: np.ndarray of downsampled time vector and main signal.
    """

    factor = int(sampling_frequency_old / sampling_frequency_new)

    #down_sampled_time = timestamp[::factor]
    down_sampled_signal = old_signal[::factor]

    return down_sampled_signal #down_sampled_time,

def welch_plot(acc, sampling_frequency, Ndivisions):

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

def stabilization_diagram(acceleration, sampling_frequency, Ndivisions, frequencies, orders):

    fig, ax =plt.subplots(figsize=(14, 6), dpi=300)

    for cluster in range(frequencies.shape[0]):
        ax.plot(frequencies[cluster], orders[cluster], marker='o', color=color_values[cluster +cluster*3])

    ax.set_ylabel("Model order")
    ax.set_xlabel('$f$ [Hz]')


    Nwindow = np.ceil(len(acceleration) / Ndivisions)  # Length of window/segment

    Nfft_pow2 = 2 ** (np.ceil(np.log2(Nwindow)))  # Next power of 2 for zero padding
    dt = 1 / sampling_frequency  # Time step

    # Call welch from scipy signal processing
    f, Sx_welch = signal.welch(acceleration, fs=1 / dt, window='hann', nperseg=Nwindow, noverlap=None, nfft=Nfft_pow2,
                               detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')

    #Sx_est = np.abs((2.0 / len(acceleration)) * np.fft.rfft(acceleration))**2*1000
    #f = np.fft.rfftfreq(acceleration.shape[0], dt)


    ax2 = ax.twinx()
    ax2.plot(f, Sx_welch, color='black', label='Welch spectrum of acceleration data', lw=0.5)
    ax2.set_ylabel("PSD")

    plt.grid()
    plt.legend()

    return fig