import numpy as np
import scipy.signal as sci
def autoSpectraParam(Fs, u, w, v_m, fitting):   
    """    
    Arguments:
    Fs      : sampling frequency
    u       : along-wind velocity data from one anemometer, without mean wind
    w       : vertical velocity data from one anemometer
    v_m     : mean wind velocity
    fitting : parameter to decide if the spectral parameters from N400 should 
              be improved by least square fitting  
    Returns:
    A_u,w    : spectral parameters for the one-point auto spectra
    lse_Au,w : least square error for A_u,w. Set to zero if fitting is not computed.
    -------------------------------------------------------------
    Function takes in the turbulence in u- and w- component for a 10 min recording, and finds the spectral parameters that gives best fit of the Kaimal auto spectra (Kaimal et al., 1972) to the spectra from data by using least squares error.
    """
    z = 54.2 # height of anemometer
    # Parameters from wind measurements
    w = w - np.mean(w) # Subtracting mean value to find zero-mean turbulence component
    sigma_u = np.std(u)
    sigma_w = np.std(w)
       
    if fitting == False:
        lse_Au = 0
        lse_Aw = 0
    else: 
        # Auto-spectra computed with Welch' method
        Nwelch = 8
        Nwindow = np.floor(np.max(np.shape(u))/Nwelch) # Length of window     
        f, S_u = sci.csd(u, u, Fs, nperseg = Nwindow)
        f, S_w = sci.csd(w, w, Fs, nperseg = Nwindow)    
        #Removing start and end values
        f = np.delete(f, [0,-1])
        S_u = np.delete(S_u, [0,-1])
        S_w = np.delete(S_w, [0,-1])      
        # Improvement of A_u,w by least square fitting
        A_val = np.arange(1,80.1,0.1) # Vector of A_i values
        lse_Au = np.zeros(np.shape(A_val)[0]) # Vector of least square errors for A_u
        lse_Aw = np.zeros(np.shape(A_val)[0]) # Vector of least square errors for A_w
        
        cnt = 0
        for A_i in A_val:
            sum_u = 0
            sum_w = 0
            for i in range(np.shape(f)[0]):
                n_uhat_i = f[i]*z/v_m
                S_u_kaimal_i = A_i*n_uhat_i/(1+1.5*A_i*n_uhat_i)**(5/3)        
                n_what_i = f[i]*z/v_m
                S_w_kaimal_i = A_i*n_what_i/(1+1.5*A_i*n_what_i)**(5/3)             
                sum_u = sum_u + (S_u[i]*f[i]/sigma_u**2 - S_u_kaimal_i)**2
                sum_w = sum_w + (S_w[i]*f[i]/sigma_w**2 - S_w_kaimal_i)**2               
            sum_u = np.sqrt(sum_u/np.shape(f)[0])
            sum_w = np.sqrt(sum_w/np.shape(f)[0])           
            lse_Au[cnt] = sum_u
            lse_Aw[cnt] = sum_w
            cnt = cnt + 1
               
        # Finding optimal value for A_u
        lse_Au_min_index = np.argmin(lse_Au)
        lse_Au = lse_Au[lse_Au_min_index]
        Au = A_val[lse_Au_min_index]      
        # Finding optimal value for A_w
        lse_Aw_min_index = np.argmin(lse_Aw)
        lse_Aw = lse_Aw[lse_Aw_min_index]
        Aw = A_val[lse_Aw_min_index]
               
    return Au, Aw, lse_Au, lse_Aw



#%% Calculation of decay coefficients for the normalised cross-spectra
def normCrossSpectraParam(Fs, u, w, dx):   
    """
    Returns the decay coefficients of the normalised cross-spectra of u and w for one single 10-min recording and chosen anemometer pairs. 
    Arguments:
    Fs   : sampling frequency
    u, w : matrices with along-wind (with mean wind) and vertical velocity data
           from closely spaced anemometers in each column. The first column is 
           the reference anemometer; the normalised cross-spectra are calculated
           between this anemometer and the others.
    dx   : vector containing the horizontal distances between the anemometers 
           in the anemometer pairs  
    Returns:
    K_u,w    : decay coefficients for the normalized cross spectra
    lse_Ku,w : least square error for K_u,w. Set to zero if fitting is not computed.
    ----------------------------------------------------------------------
    Function returns the fitted decay coefficients of the normalised cross-spectra of u and w for one single 10 min recording and chosen anemometer pairs. The fit is done by using least square error. The expression that is fitted is Davenport's expression(Davenport, 1961).
    """ 
    # Parameters from N400
    Ku = 10.0
    Kw = 6.5 
    # Parameters needed to estimate Cuu and Cww with Welch method
    Nwelch = 8
    Nwindow = np.floor(np.max(np.shape(u))/Nwelch) # Length of window       
    f = sci.csd(u[:,0], u[:,0], Fs, nperseg = Nwindow)[0] # Frequency axis  
    # Estimation of normalised cross-spectra and reduced frequency for each anemometer pair
    Cuu = np.zeros([np.shape(f)[0], np.shape(u)[1]-1])
    Cww = np.zeros([np.shape(f)[0], np.shape(u)[1]-1])
    f_red = np.zeros([np.shape(f)[0], np.shape(u)[1]-1])
    for i in range(np.shape(u)[1]-1):      
        Suu = sci.csd(u[:,0]-np.mean(u[:,0]), u[:,i+1]-np.mean(u[:,i+1]), Fs, nperseg = Nwindow)[1] # Cross-spectrum
        Su1 = sci.csd(u[:,0]-np.mean(u[:,0]), u[:,0]-np.mean(u[:,0]), Fs, nperseg = Nwindow)[1] # Auto-spectrum of u1
        Su2 = sci.csd(u[:,i+1]-np.mean(u[:,i+1]), u[:,i+1]-np.mean(u[:,i+1]), Fs, nperseg = Nwindow)[1] # Auto-spectrum of u2
        Cuu[:,i] = Suu / np.sqrt(Su1*Su2)
        
        Sww = sci.csd(w[:,0]-np.mean(w[:,0]), w[:,i+1]-np.mean(w[:,i+1]), Fs, nperseg = Nwindow)[1] # Cross spectrum
        Sw1 = sci.csd(w[:,0]-np.mean(w[:,0]), w[:,0]-np.mean(w[:,0]), Fs, nperseg = Nwindow)[1] # Auto-spectrum of w1
        Sw2 = sci.csd(w[:,i+1]-np.mean(w[:,i+1]), w[:,i+1]-np.mean(w[:,i+1]), Fs, nperseg = Nwindow)[1] # Auto-spectrum of w2
        Cww[:,i] = Sww / np.sqrt(Sw1*Sw2)
        
        f_red[:,i] = f * dx[i]/(np.mean([np.mean(u[:,0]), np.mean(u[:,i+1])])) # Reduced frequency
    ### Improvement of K_u,w by least square fitting   
    K_val = np.arange(1,25.1,0.1) # Vector of K_i values
    lse_Ku = np.zeros(np.shape(K_val)[0]) # Vector of least square errors for K_u
    lse_Kw = np.zeros(np.shape(K_val)[0]) # Vector of least square errors for K_w
    
    cnt = 0
    for K_i in K_val:
        sum_u = 0
        sum_w = 0
        for p in range(np.shape(dx)[0]):
            for i in range(np.shape(f)[0]):
                C_i = np.exp(-K_i*f_red[i, p]) 
                sum_u = sum_u + (Cuu[i, p] - C_i)**2
                sum_w = sum_w + (Cww[i, p] - C_i)**2         
        sum_u = np.sqrt(sum_u/(np.shape(f)[0]*np.shape(dx)[0]))
        sum_w = np.sqrt(sum_w/(np.shape(f)[0]*np.shape(dx)[0]))     
        lse_Ku[cnt] = sum_u
        lse_Kw[cnt] = sum_w
        cnt = cnt + 1
  
    # Finding optimal value for K_u
    lse_Ku_min_index = np.argmin(lse_Ku)
    lse_Ku = lse_Ku[lse_Ku_min_index]
    Ku = K_val[lse_Ku_min_index]   
    # Finding optimal value for K_w
    lse_Kw_min_index = np.argmin(lse_Kw)
    lse_Kw = lse_Kw[lse_Kw_min_index]
    Kw = K_val[lse_Kw_min_index]
               
    return Ku, Kw, lse_Ku, lse_Kw