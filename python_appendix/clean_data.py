import numpy as np
from scipy.interpolate import interp1d

def remove_error(data, fs, status):
    '''
    Arguments :
    data   : time series
    fs     : sampling frequency
    status : vector of status codes from the sensor 
    Returns :
    newData     : cleaned time series
    error_ratio : amount of error values in time series
    ---------------------------------------------------------------
    Function returns the clean time series with a time vector. Time
    is assumed to start from zero. The new series have the same sampling rate as the input
    Linear interpolation is conducted to preserve the sampling rate
    NB! Will result in error if the first or last sample is an error value (time interpolation will fail) 
    '''
    #Generate time vector
    l = np.max(np.shape(data))
    t = np.arange(0,l)*1/fs
    t1 = t #Store initial time vector 
    #Deleting the values that have error
    error_ind = np.where(np.logical_and(status!='00', status!='0A'))[0]
    #Apply linear interpolation if there are error values
    if np.size(error_ind)!=0:
        data = np.delete(data, error_ind, axis=0)
        t = np.delete(t, error_ind)
        newData = np.zeros((l, 3))
        for i in range(3):
            if np.size(data[:,i])==0:
                newData[:,i] = np.zeros((l, np.shape(data[:,i])[1]))
                error_ratio = 1
            else:
                newData[:,i] = interp1d(t, data[:,i])(t1)
                error_ratio = np.size(error_ind)/l
    else:
        newData = data
        error_ratio = 0
        
    return newData, error_ratio
    
def remove_std(data, fs, x):
    '''
    Arguments:
    data    : time series
    x       : discard values greater than x*std in an absolute manner
    fs      : sampling frequency
    Returns:
    datn    : time series
    stdrig  : bool proclaiming if there were any values exceeding limit
    ------------------------
    Function discards values larger than x times the standard deviation in an 
    absolute sense. Returns a clean time series with same sampling rate as input.
    Linear interpolation is conducted to preserve the sampling rate.
    
    Function is based on code written by Aksel Fenerci
    '''
    #Generate time vector
    l = np.max(np.shape(data))
    t = np.arange(0, l)*1/fs
    t1 = t
    
    ind = []
    datn = np.zeros((l, 3))
    for i in range(3):
        dummy = np.argwhere(data[:,i] > np.mean(data[:,i]) +x *np.std(data[:,i]))
        dummy2 = np.argwhere(data[:,i] < np.mean(data[:,i]) - x*np.std(data[:,i]))
        ind = np.concatenate((dummy, dummy2))
        if np.size(ind)!=0:
            stdtrig = True
        else:
            stdtrig = False       
        if stdtrig: 
            ind = np.sort(ind, axis=0)
            if ind[0] == 0:
                ind = np.delete(ind, 0)
                dat = np.delete(data[:,i], ind)
                t = np.delete(t1, ind)
                dat[0] = np.mean(data[:,i]) +x *np.std(data[:,i])
            elif ind[-1] == l-1:
                ind = np.delete(ind, -1)
                dat = np.delete(data[:,i], ind)
                t = np.delete(t1, ind)
                dat[-1] = np.mean(data[:,i]) +x *np.std(data[:,i])
            else:
                dat = np.delete(data[:,i], ind)
                t = np.delete(t1, ind)
            datn[:,i] = interp1d(t, dat)(t1)
        else:
            datn[:,i] = data[:,i]
            
    return datn, stdtrig
       
def circArray(data, dataRange):
    '''
    Arguments:
    data      : time series
    dataRange : range where the value of data is valid
    Returns:
    dataCirc : corrected data
    ------------------------------------------------
    Function translates circular array (typically angles).
    
    Function based on code written by Knut Andreas Kvaale
    '''
    dataCirc = data
    span = np.max(dataRange) - np.min(dataRange)    
    dataCirc[data>np.max(dataRange)] = dataCirc[data>np.max(dataRange)] - span
    dataCirc[data<np.min(dataRange)] = span + dataCirc[data<np.min(dataRange)]    
    
    return dataCirc