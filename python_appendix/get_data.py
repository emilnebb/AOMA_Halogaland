import numpy as np
import measurementFunctions as mf
import lowPass_DownSample as ld
import clean_data as cd
import windCharacteristics as wc
import time_synchronisation as ts

def readAcc(anodeName, tdmsFile, interval, t_master, downFreq):
    """
    Arguments:
    anodeName : name of logger box
    tdmsFile  : one .tdms-file, containing raw data from 8 hour of recordings 
                from one data logger
    interval  : chosen length of intervals in minutes
    t_master  : master time vector, for time synchronization
    downFreq  : frequency that the signal is downsampled to  
    Returns:
    acc_y     : processed horizontal acceleration data, n_intervals*n_samples
    acc_z     : processed vertical acceleration data, n_intervals*n_samples
    acc_theta : processed torsional acceleration data, n_intervals*n_samples
    ----------------------------------------------------------------------
    Function extracts processed acceleration data from one logger box for eight hours of data
    """    
    # Constants
    g = 9.82        # Gravity constant 
    B = 18.6        # Width of bridge deck
    Fs = 64         # Sampling frequency
    N = 10          # Order of the low pass filter
    acc_names = ['-1x', '-1y', '-1z', '-2x', '-2y', '-2z'] 
    # Retrieving data from file
    acc_data = tdmsFile['acceleration_data']   
    length = np.max(np.shape(acc_data['timestamp'][:])) # Length of acceleration series    
    # Fixing correct units
    acc_raw = np.zeros((length, 6))
    for i in range(6):
        conversion_factor = float(acc_data[anodeName + acc_names[i]].properties['conversion_factor'])
        acc_raw[:,i] = acc_data[anodeName + acc_names[i]][:] / conversion_factor * g    
    # Time synchronisation
    acc_sync = np.zeros((int(length/(Fs/32)), 6))
    for i in range(6):
        acc_sync[:,i] = ld.lowPassAndDownSample(acc_raw[:,i] - np.mean(acc_raw[:,i]), 0.5*32, N, 32, Fs) + np.mean(acc_raw[:,i])
    stamp = acc_data['timestamp'][:]
    stamp_sync = ld.lowPassAndDownSample(acc_raw[:,0] - np.mean(acc_raw[:,0]), 0.5*32, N, 32, Fs, stamp)[0]
    acc_sync_fixed = np.zeros((np.shape(t_master)[0], 6))
    for i in range(6):
        acc_sync_fixed[:,i] = ts.time_interpolation(acc_sync[:,i], stamp_sync, t_master)
    
    Fs = 32 # New sampling frequency
    length = np.shape(acc_sync_fixed)[0] # New length of acceleration series  
    # Lowpass and down sample
    acc = np.zeros((int(length/(Fs/downFreq)), 6))
    for i in range(6):
        acc[:,i] = ld.lowPassAndDownSample(acc_sync_fixed[:,i] - np.mean(acc_sync_fixed[:,i]), 0.5*downFreq, N, downFreq, Fs) + np.mean(acc_sync_fixed[:,i])
    dt_new = 1 / downFreq 
    t_new = np.arange(0, np.max(np.shape(acc))) * dt_new   
    length_new = np.max(np.shape(acc))   
    # Preparations for division in intervals
    ints = t_new[-1] / 60 / interval # Number of intervals     
    if ints < 1: # If number of intervals is less than one
        ints = 1        
    if ints % 1 != 0: # If number of intervals is a decimal number
        ints = int(np.ceil(ints)) - 1           
    l2 = interval * 60 * downFreq * ints + ints # New length               
    acc = np.delete(acc, np.arange(l2, length_new, 1), axis=0)    
    # Dividing into intervals
    p = int(l2 / ints)
    pp = p
    c = 0
    # Initializing acceleration matrices
    acc_y = np.zeros([ints, p])
    acc_z = np.zeros([ints, p])
    acc_theta = np.zeros([ints, p])

    acc_1x = np.zeros([ints, p])
    acc_1y = np.zeros([ints, p])
    acc_1z = np.zeros([ints, p])
    acc_2x = np.zeros([ints, p])
    acc_2y = np.zeros([ints, p])
    acc_2z = np.zeros([ints, p])

    #print(acc_y.shape)
    #print(acc.shape)

    for i in range(ints): 
        acc_trans = mf.transform(acc[c:pp, 0:3], acc[c:pp, 3:6], B)
        #print(acc_trans.shape)
        acc_1x[i, :] = acc[c:pp, 0]
        acc_1y[i, :] = acc[c:pp, 1]
        acc_1z[i, :] = acc[c:pp, 2]
        acc_2x[i, :] = acc[c:pp, 3]
        acc_2y[i, :] = acc[c:pp, 4]
        acc_2z[i, :] = acc[c:pp, 5]
        c = c + p
        pp = pp + p
 
    return acc_1x, acc_1y, acc_1z, acc_2x, acc_2y, acc_2z
#%% Function for extracting prosessed wind-data from one anemometer

def readWind(anemometerName, tdmsFile, interval, t_master, downFreq):
    """
    Arguments:
    anemometerName : name of anemometer from the tdmsFile
    tdmsFile : one .tdms-file, containing raw data from 8 hour of recordings 
               from one logger box
    interval : chosen length of intervals in minutes
    t_master : master time axis, for time synchronization
    downFreq : frequency that the signal is down sampled to
    Returns:
    u : processed wind data; along wind direction, n_samples*n_intervals
    w : processed wind data; vertical direction, n_samples*n_intervals
    -------------------------------------------------------------------
    Function extracts wind data from one anemometer and processes it.
    """  
    Fs = 32         # Sampling frequency
    N = 10          # Order of the low pass filter
    #Retrieving data from file
    wind = tdmsFile[anemometerName]
    status = wind['status_code'][:]  
    direction = wind['direction'][:]
    magnitude = wind['magnitude'][:]
    vertical = wind['vertical_velocity'][:]
    # Combining wind data into one big matrix
    wind_data = np.zeros((np.max(np.shape(direction)), 3))
    wind_data[:,0] = direction
    wind_data[:,1] = magnitude
    wind_data[:,2] = vertical  
    # Removing error values
    newData, error_ratio = cd.remove_error(wind_data, Fs, status)  
    # Time synchronization
    stamp = wind['timestamp'][:]  
    data_sync = np.zeros((np.shape(t_master)[0], 3))
    for i in range(3):
        data_sync[:,i] = ts.time_interpolation(newData[:,i], stamp, t_master)
    # Clean data
    newData, stdtrig = cd.remove_std(data_sync, Fs, 6)  
    # Fixing spike in directional data
    newData[:,0]= cd.circArray(newData[:,0], [-180 + np.mean(newData[:,0]), 180 + np.mean(newData[:,0])])
    # Resampling the data
    if Fs != downFreq:
        length = np.max(np.shape(newData))
        data_rs = np.zeros((int(length/(Fs/downFreq)), 3))
        for i in range(3):
            data_rs[:,i] = ld.lowPassAndDownSample(newData[:,i] - np.mean(newData[:,i]), 0.5*downFreq, N, downFreq, Fs, False) + np.mean(newData[:,i])
    else:
        data_rs = newData
    # Obtaining the turbulence components
    V, u, v, w, meandir = wc.transform_uvw(np.real(data_rs), downFreq, interval)
    # Applying highpass filter
    for i in range(np.min(np.shape(u))):
        u[:,i] = ld.filter_data(u[:,i] - np.mean(u[:,i]), 1/300, 2, downFreq, 'highpass') + np.mean(u[:,i])
        w[:,i] = ld.filter_data(w[:,i] - np.mean(w[:,i]), 1/300, 2, downFreq, 'highpass') + np.mean(w[:,i])

    return u, w, V, meandir

def getCharacteristics(tdms_file, node_name, anemometer, t_master, downFreq):
    '''
    Arguments:
    filename   : a tdms-file containing 8 hours of data
    node_name  : a string specifiying which node to look at
    anemometer : string containing the name of the chosen anemometer
    t_master   : master time axis, for time synchronisation
    downFreq   : downFreq : frequency that the signal is down sampled to 
    Returns:
    V               : mean wind speed
    sigma_u,w       : standard deviation for turbulence components
                      of 10 minutes
    I_u,w           : turbulence intensity
    u, w            : turbulence components
    sigma_y,z,theta : standard deviation of acceleration
    --------------------------
    Function goes through one eight-hour file and extracts wind and acceleration characteristics for intervals of 10 minutes. The file that is
    given as argument should be read in advance.
    '''
    # Acceleration
    acc_y, acc_z, acc_theta = readAcc(node_name, tdms_file, 10, t_master, downFreq)
    sigma_y = np.std(acc_y, axis=1)
    sigma_z = np.std(acc_z, axis=1)
    sigma_theta = np.std(acc_theta, axis=1)
    # Wind
    u, w, V, meandir = readWind(anemometer, tdms_file, 10, t_master, downFreq) 
    _, _, I_u, I_w = wc.windProperties(u, w - np.mean(w, axis=0), V)
    
    return V, meandir, u, w, I_u, I_w, sigma_y, sigma_z, sigma_theta 