import numpy as np
from scipy import signal

def low_pass(old_signal: np.ndarray, sampling_frequency, cutoff_frequency, filter_order) -> np.ndarray:
    """
    Apply Butterworth filter as lowpass filter
    :param signal: np.ndarray of signal
    :param sampling_frequency: int sampling frequency of the signal
    :param cutoff_frequency: int cutoff frequency, which frequency of higher content will be removed
    :param filter_order: int hyperparameter to the Butterworth filter
    :return: np.ndarray filtered signal
    """
    sos = signal.butter(filter_order, cutoff_frequency, "low",
                        fs=sampling_frequency, output='sos')

    filtered_signal = signal.sosfilt(sos, old_signal)

    return filtered_signal

def downsample(timestamp: np.ndarray, old_signal: np.ndarray, sampling_frequency_new) -> np.ndarray:
    """
    Downsampling of main signal with corresponding time vector.
    :param timestamp: np.ndarray of timestamps of the main signal
    :param old_signal: np.ndarray of main signal
    :param sampling_frequency_new: int
    :return: np.ndarray of downsampled time vector and main signal.
    """
    dt = timestamp[1] - timestamp[0]
    sampling_frequency_old = 1/dt
    factor = int(sampling_frequency_old / sampling_frequency_new)

    down_sampled_time = timestamp[::factor]
    down_sampled_signal = old_signal[::factor]

    return down_sampled_time, down_sampled_signal