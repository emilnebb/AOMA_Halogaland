import numpy as np
from scipy.interpolate import interp1d
def time_master(tdms_files, wind_nodes):
    '''
    Arguments:
    tdms_files      : list of tdms_file
    wind_nodes      : list of strings containing the common part of anemometer name belonging
                      to considered anode(s). If looking at all nodes, wind_nodes is full list with
                      'xx' at indices where there are no anemometers. If looking at one anode, wind_nodes
                      only contains the specific anodes
    Returns:
    t_master     : master time vector given in seconds
    -----------------------------------------
    Function takes in a list of tdms-objects, and searches for the one
    that starts sampling last and the one that ends sampling first, and computes
    the master time vector from these. The search includes all anemometers.
    '''
    start_time = 0 #Initial value for start of master time vector
    end_time = 1e100 #Initial value for end of master time vector 
    
    #Finding start time and end time
    for i in range(len(tdms_files)):
        temp_start = tdms_files[i]['acceleration_data']['timestamp'][0]
        temp_end = tdms_files[i]['acceleration_data']['timestamp'][-1]
        if temp_start > start_time:
            start_time = temp_start
        if temp_end < end_time:
            end_time = temp_end
        anemometers = [s for s in list(tdms_files[i]) if wind_nodes[i] in s]
        for k in range(len(anemometers)):
            temp_start = tdms_files[i][anemometers[k]]['timestamp'][0]
            temp_end = tdms_files[i][anemometers[k]]['timestamp'][-1]
            if temp_start > start_time:
                start_time = temp_start
            if temp_end < end_time:
                end_time = temp_end
    
    t_master = np.arange(np.ceil(start_time*1e-9), np.floor(end_time*1e-9), 1/32)
    return t_master

def time_interpolation(data, timestamp, t):
    '''
    Arguments:
    data        : the data to be interpolated
    timestamp   : timestamp corresponding to data, given in nanoseconds
    t           : master time vector, given in seconds   
    Returns:
    datn        : time synchronised data
    ---------------------------
    Function interpolates the data to a master time vector
    '''
    t1 = t
    t = timestamp*1e-9   
    datn = interp1d(t, data)(t1)
    return datn