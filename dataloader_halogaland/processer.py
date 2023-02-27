import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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