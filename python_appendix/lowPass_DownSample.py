import numpy as np
from scipy import signal
def filter_data(s, Wn, N, Fs, ftype):
    '''
    Arguments:
    s        : signal, 1D-array
    Wn       : cutoff frequency
    N        : order of the filter
    Fs       : sampling frequency of signal
    ftype    : type of filter {'lowpass', 'highpass'}
    Returns:
    filtered : filtered signal 
    ----------------------------------------------------------------------------    
    Takes a signal and applies a filter using Butterworth filter of order N.
    '''
    #Normalizes the cutoff frequency, with 1 being the Nyquist frequency 
    normWn = 2*Wn/Fs  
    #Filter coefficients 
    sos = signal.butter(N, normWn, output='sos', btype=ftype) 
    #Forward-backward digital filter
    filtered = signal.sosfiltfilt(sos, s)
    
    return filtered

def lowPassAndDownSample(s, Wn, N, downFreq, Fs, t=[]):
    '''
    Arguments:
    s        : signal, 1D-array
    Wn       : cutoff frequency, 40 % of the downFreq
    N        : order of the low pass filter
    downFreq : the frequency that the signal is down sampled to
    Fs       : sampling frequency of signal
    t        : optional argument, time vector to be downsampled  
    Returns:
    newSignal : down-sampled signal   
    ----------------------------------------------------------------------------    
    Takes a signal and applies low pass filter using Butterworth filter, then down samples
    the signal. If t is given, the function also outputs a resampled time vector
    '''   
    #Low pass filter
    filtered = filter_data(s, Wn, N, Fs, 'lowpass')       
    #%%Down sampling by decimation
    factor = (Fs/downFreq)
    samples = np.shape(filtered)[0]  #Number of samples in original signal 
    n = int(samples/factor) #number of samples in new signal
    
    if len(t)!=0:
        time_out = True
        t_new = np.zeros(n)
    else:
        time_out = False     
    newSignal = np.zeros(n)
    for i in range(n):
        newSignal[i] = filtered[i*int(factor)]
        if time_out:
            t_new[i] = t[i*int(factor)]

    if time_out:
         return t_new, newSignal
    else:
         return newSignal   