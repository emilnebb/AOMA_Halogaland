import numpy as np

def angmean(angles):
    '''
    Arguments :
    angles : array of angles
    Returns :
    mean_angle : mean angle of the array 
    ---------------------------
    Function establishes mean of an array of angles
    
    Function based on code written by Knut Andreas Kvaale
    '''
    mean_angle = np.arctan2(np.nanmean(np.sin(angles*np.pi/180)), np.nanmean(np.cos(angles*np.pi/180)))*180/np.pi 
    #Adjust the values to lie between 0 and 360 degrees
    if mean_angle<0:
        mean_angle = mean_angle + 360
    return mean_angle

def transform_uvw(data, Fs, interval, varargin=[]):
    '''
    Arguments :
    data     : a nx3 matrix, where data[:,0] is direction, data[:,1] is magnitude
               in horizontal direction and data[:,2] is the vertical velocity               
    Fs       : sampling frequency
    interval : time interval in minutes
    varargin : list of additional arguments, varargin[0] is the index of which
               ten-minute interval that is of interest, and varargin[1] is a 
               string defining whether to detrend or not             
    Returns :
    V       : mean wind velocity
    u       : turbulence component in along-wind direction
    v       : turbulence component in across-wind direction
    w       : turbulence component in vertical direction, mean wind included
    meandir : mean direction of the wind
    ----------------------------------------------------
    Function transforms wind data to uvw coordinates from the polar coordinates
    
    Function based on code written by Aksel Fenerci
    '''
    detrend_opt = []
    ind = [] 
    if np.size(varargin)!=0:
        ind = [varargin[0]]
        if np.size(varargin)>1:
            detrend_opt = varargin[1] 
    if detrend_opt!='on':
        detrend_opt = 'off' #default
    # Finding number of intervals in data
    l = np.max(np.shape(data))
    t = np.arange(0,l)*1/Fs  
    ints = t[-1]/60/interval
    if ints < 1:
        ints = 1      
    if ints%1 !=0:
        ints = int(np.ceil(ints)) - 1       
    l2 = interval*60*Fs*ints + ints #The index of the last element in the last whole interval
            
    data = np.delete(data, np.arange(l2,l, 1), axis=0) #Deleting the excess data that does not fit in an interval    
    p = int(l2/ints)
    pp = p
    c = 0   
    #Initializing matrices
    meandir = np.zeros(ints) 
    phi = np.zeros((pp,ints))
    V_alpha=  np.zeros((pp, ints))
    V_beta = np.zeros((pp, ints))
    V = np.zeros(ints)
    u = np.zeros((pp,ints))
    v = np.zeros((pp, ints))
    w = np.zeros((pp, ints))
    if np.any(np.isnan(data)):
        meandir = np.nan
        V = np.nan
        u = np.nan
        v = np.nan
        w = np.nan
    else:
        if np.size(ind)==0:
            for i in range(ints):
                if ~np.any(np.isnan(np.squeeze(data[c:pp,:]))):
                    meandir[i] = angmean(data[c:pp,0])
                    phi[:,i] = (data[c:pp,0] - meandir[i])*np.pi/180
                    V_alpha[:,i] = data[c:pp,1]*np.cos(phi[:,i])
                    V_beta[:,i] = data[c:pp,1]*np.sin(phi[:,i])
                    V[i] = np.nanmean(V_alpha[:,i])
                    u[:,i] = V_alpha[:,i] - V[i]
                    v[:,i] = V_beta[:,i]
                    if detrend_opt =='on':
                        w[:,i] = data[c:pp,2] - np.nanmean(data[c:pp,2])
                    elif detrend_opt == 'off':
                        w[:,i] = data[c:pp,2]
                else:
                    V[i] = np.nan
                    meandir[i] = np.nan
                    u[:,i] = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))
                    w[:,i] = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))
                    w[:,i] = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))            
                c = c + p
                pp = pp + p       
        else:      #Transformation for one particular interval  
            c = c + p*(ind)
            pp = pp + p*(ind)         
            if ~np.any(np.isnan(np.squeeze(data[c:pp,:]))):
                meandir = angmean(data[c:pp,0])
                phi = (data[c:pp,0] - meandir)*np.pi/180
                V_alpha = data[c:pp,1]*np.cos(phi)
                V_beta = data[c:pp,1]*np.sin(phi)
                V = np.nanmean(V_alpha)
                u = V_alpha - V
                v = V_beta             
                if detrend_opt == 'on':
                    w = data[c:pp,2] - np.nanmean(data[c:pp,2])
                elif detrend_opt == 'off':
                    w = data[c:pp,2]             
            else:
                V = np.nan
                meandir = np.nan
                u = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))
                v = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))
                w = np.nan*np.arange(0, np.max(np.shape(data[c:pp,1])))
                
    return V, u, v, w, meandir    
        
def windProperties(u, w, V):
    """
    Arguments :
    u : turbulence in along wind direction for one anemometer
    w : turbulence in vertical direction for one anemometer
    V : mean wind speed   
    Returns:
    sigma_u : standard deviation of along-wind turbulence
    sigma_W : standard deviation of vertical turbulence
    Iu      : turbulence intensity in along wind direction
    Iw      : turbulence intensity in vertical direction
    ------------------------------------------
    Function finds standard deviation and turbulence intensity of the wind. The turbulence given as input is without mean value.
    """
    # Standard deviation
    sigma_u = np.std(u, axis=0)
    sigma_w = np.std(w, axis=0) 
    # Turbulence intensity
    I_u = sigma_u/V
    I_w = sigma_w/V
    
    return sigma_u, sigma_w, I_u, I_w
