import numpy as np
from scipy import signal


def low_pass(old_signal: np.ndarray, sampling_frequency, cutoff_frequency, filter_order) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the input signal.

    Args:
        old_signal (np.ndarray): The input signal to be filtered.
        sampling_frequency (float): The sampling frequency of the input signal.
        cutoff_frequency (float): The cutoff frequency of the low-pass filter.
        filter_order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The filtered signal after applying the low-pass filter.
    """

    sos = signal.butter(filter_order, cutoff_frequency, btype='lowpass',
                        fs=sampling_frequency, output='sos')

    filtered_signal = signal.sosfilt(sos, old_signal)

    return filtered_signal


def downsample(sampling_frequency_old, old_signal: np.ndarray, sampling_frequency_new) -> np.ndarray:
    """
    Downsample the input signal by a given factor.

    Args:
        sampling_frequency_old (float): The original sampling frequency of the input signal.
        old_signal (np.ndarray): The input signal to be downsampled.
        sampling_frequency_new (float): The desired sampling frequency of the downsampled signal.

    Returns:
        np.ndarray: The downsampled signal.
    """

    factor = int(sampling_frequency_old / sampling_frequency_new)

    down_sampled_signal = old_signal[::factor]

    return down_sampled_signal