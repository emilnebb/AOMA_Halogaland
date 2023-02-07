#%% Import necessary modules
from scipy.interpolate import interp1d
import AbaqusInputToBuffetingResponse as inp
from csv import writer
import os
import numpy as np
def RMS(U, I_u, I_w, Au, Aw, Ku, Kw, x_r, spectra_type, filename):   
    """
    Arguments:
    U : mean wind velocity 
    I_u,w        : turbulence intensity of along-wind and vertical turbulence component
    A_u,w        : spectral parameters for the one-point auto spectra
    K_u,w        : decay coefficients for the normalized cross spectra
    x_r          : index in x-vector for position along the bridge
    spectra_type : 'kaimal' / 'N400'
    filename     : name of the file that the outputs should be saved to    
    Returns:
    rms_tot : vector of root mean square of acceleration response for y, z and theta
    ms      : array of mean square of acceleration response for y, z and theta, 
              from each mode
    ----------------------------------------------------------------------
    Function calculates the buffeting response using either a kaimal type of spectra, or the N400 spectra.  
    """   
    ##### Constants / parameters   
    rho = 1.25 # Air density 
    B = 18.6 # Width of bridge cross section 
    D = 3 # Height of bridge cross section     
    # Calculation of integral length scale
    L1 = 100 # Reference length scale
    z1 = 10 # Reference height
    z = 54.2 # height of anemometer 
    xLu = L1 * (z/z1)**0.3
    xLw = 1/12 * xLu    
    if spectra_type == 'kaimal':
        xLu = z
        xLw = z
    # Retrieving modal parameters from Abaqus
    nodes, phi, w_n, Mi_t = inp.modalParam(numberOfModes=56)
    phi = np.transpose(phi, [0,2,1])
    #Interpolation of nodal coordinates and mode shapes
    L = np.max(nodes) - np.min(nodes)
    x = np.linspace(np.ceil(-L/2), np.floor(L/2), 281) # Coordinates
    phi = np.array(interp1d(nodes, phi, axis=1)(x))
    
    ksi = 0.005 # Damping ratio for each mode, set to 0.5%    
    omega = np.logspace(-3, 0, 100)*2*np.pi # Vector of frequencies [rad/s]   

    Cd_bar = 0.793; Cl_bar = -0.353; Cm_bar = -0.0149; Cd_prime = -1.0842; Cl_prime = 3.420; Cm_prime = 1.0625 # From wind tunnel test at NTNU
    Bq = rho*U*B/2 * np.array([[2*(D/B)*Cd_bar, (D/B)*Cd_prime-Cl_bar], [2*Cl_bar, Cl_prime+(D/B)*Cd_bar], [2*B*Cm_bar, B*Cm_prime]])
    
    sigma_u = I_u * U
    sigma_w = I_w * U   
    
    ##### Calculations #####    
    # Calculating the spectral density of the generalized load through a double for-loop
    print('(1) Calculation of spectral density for the generalized loads.')   
    Su = 1/(2*np.pi) * sigma_u**2 * xLu / U * Au / (1 + 1.5 * Au * (omega*xLu/(2*np.pi*U)))**(5/3) # One-point auto-spectrum for u
    Sw = 1/(2*np.pi) * sigma_w**2 * xLw / U * Aw / (1 + 1.5 * Aw * (omega*xLw/(2*np.pi*U)))**(5/3) # One-point auto-spectrum for w  
    S_Q_red = np.zeros([np.shape(omega)[0], np.shape(w_n)[0]])
    dxdx = np.abs(np.array([x]) - np.array([x]).T)
    for i in range(np.shape(w_n)[0]):
        print('    Calculating for mode nr. ' + str(i+1) + ' out of ' + str(np.shape(w_n)[0]))
        for w in range(np.shape(omega)[0]):           
            Suu = Su[w] * np.exp(-Ku*omega[w]*dxdx/(2*np.pi*U))
            Sww = Sw[w] * np.exp(-Kw*omega[w]*dxdx/(2*np.pi*U))
            Int_1 = (np.array([phi[0,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[0,0] + np.array([phi[1,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[1,0] + np.array([phi[2,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[2,0]) * Suu * Bq[0,0] + \
                    (np.array([phi[0,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[0,1] + np.array([phi[1,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[1,1] + np.array([phi[2,:,i]]).T @ np.array([phi[0,:,i]]) * Bq[2,1]) * Sww * Bq[0,1]           
            Int_2 = (np.array([phi[0,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[0,0] + np.array([phi[1,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[1,0] + np.array([phi[2,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[2,0]) * Suu * Bq[1,0] + \
                    (np.array([phi[0,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[0,1] + np.array([phi[1,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[1,1] + np.array([phi[2,:,i]]).T @ np.array([phi[1,:,i]]) * Bq[2,1]) * Sww * Bq[1,1]           
            Int_3 = (np.array([phi[0,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[0,0] + np.array([phi[1,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[1,0] + np.array([phi[2,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[2,0]) * Suu * Bq[2,0] + \
                    (np.array([phi[0,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[0,1] + np.array([phi[1,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[1,1] + np.array([phi[2,:,i]]).T @ np.array([phi[2,:,i]]) * Bq[2,1]) * Sww * Bq[2,1]            
            Int = Int_1 + Int_2 + Int_3
            S_Q_red[w,i] = np.trapz(np.trapz(Int,x), x)

            
    ## Refining the w-axis for the generalized loads by interpolation  
    print('(2) Refining the frequency axis')    
    omega_ref = np.linspace(0.001, 1, 1000)*2*np.pi
    S_Q = interp1d(omega, S_Q_red, axis=0)(omega_ref)
    
    ## Calculating generalized damping and stiffness
    print('(3) Calculating generalized damping and stiffness')    
    Ci_t = np.zeros(np.shape(w_n)[0]) # Generalized damping 
    Ki_t = np.zeros(np.shape(w_n)[0]) # Generalized stiffness 
    for i in range(np.shape(w_n)[0]):
        Ci_t[i] = 2*Mi_t[i]*w_n[i]*ksi
        Ki_t[i] = w_n[i]**2*Mi_t[i] 
        
    ## Calculating aerodynamic damping and stiffness
    print('(4) Calculating aerodynamic damping and stiffness')    
    Cae = np.zeros([np.shape(omega_ref)[0], 3, 3]) # Aerodynamic damping
    Kae = np.zeros([np.shape(omega_ref)[0], 3, 3]) # Aerodynamic stiffness    
    i = 0
    for w in omega_ref:
        K = w*B/U # Reduced frequency
        v_hat = 1 / K           
        # Aerodynamic derivatives
        if v_hat <= 1.35:
            v_hat = 1.35
        elif v_hat >= 17:
            v_hat = 17        
        P4s = (-8.73239e-05*v_hat**2 + 0.00220737*v_hat + 0.0263768)/K**2
        H4s = (-0.00143984*v_hat**2 + 0.0325336*v_hat + -0.145406)/K**2
        A3s = (-0.000826889*v_hat**2 + 0.0192049*v_hat + 0.926908)/K**2
        P1s = (0.00338829*v_hat**2 + -0.070592*v_hat + 0.00153207)/K
        H1s = (0.00530354*v_hat**2 + -0.116064*v_hat + -2.19524)/K
        A2s = (0.0017332*v_hat**2 + -0.0452137*v_hat + -0.157346)/K
        
        Cae[i, :, :] = (rho*B**2/2)*w * np.array([[P1s, 0, 0], [0, H1s, 0], [0, 0, B**2*A2s]]) 
        Kae[i, :, :] = (rho*B**2/2)*w**2 * np.array([[P4s, 0, 0], [0, H4s, 0], [0, 0, B**2*A3s]]) 
        i = i + 1

    ## Calculating the response
    print('(5) Calculating the response')     
    ms = np.zeros([3, np.shape(w_n)[0]]) # Mean square of acceleration response for y, z and theta
    # Looping through all frequencies
    S_rdd = np.zeros([3, np.shape(omega_ref)[0]]) # Matrix with auto-spectra for acc. resp. in y, z, theta
    # Looping through each mode
    for i in range(np.shape(w_n)[0]):
        print('    Calculating for mode nr. ' + str(i+1) + ' out of ' + str(np.shape(w_n)[0]))        
        j = 0
        for w in omega_ref:    
            Ci_aet = np.trapz(phi[0,:,i]**2*Cae[j,0,0] + phi[1,:,i]**2*Cae[j,1,1] + phi[2,:,i]**2*Cae[j,2,2], x) # Generalized aerodynamic damping 
            Ki_aet = np.trapz(phi[0,:,i]**2*Kae[j,0,0] + phi[1,:,i]**2*Kae[j,1,1] + phi[2,:,i]**2*Kae[j,2,2], x) # Generalized aerodynamic stiffness             
            Hi_t = 1/(-Mi_t[i]*w**2 + (Ci_t[i] - Ci_aet)*1j*w + (Ki_t[i] - Ki_aet)) # Generalized frequency response function
            Si_eta = np.abs(Hi_t)**2 * S_Q[j, i] # Auto-spectrum for generalized response           
            Si_r = phi[:, x_r, i]**2 * Si_eta # Auto-spectra for disp. resp. in y, z, theta
            Si_rdd = w**4 * Si_r # Auto-spectra for acc. resp. in y, z, theta           
            S_rdd[:, j] = Si_rdd           
            j = j + 1                   
        ms[0, i] = np.trapz(S_rdd[0, :], omega_ref)
        ms[1, i] = np.trapz(S_rdd[1, :], omega_ref)
        ms[2, i] = np.trapz(S_rdd[2, :], omega_ref)
    rms_tot = np.sqrt(np.array([sum(ms[0]), sum(ms[1]), sum(ms[2])]))
      
    # Saving vibration_data
    save_path = os.getcwd()+ '\Results' + '\\Predicted'
    file_data = np.array([U, rms_tot[0], rms_tot[1], rms_tot[2]]).T
    with open(save_path + filename + '.csv', 'a', newline='') as file:
        file_write = writer(file)
        file_write.writerow(file_data)
        file.close()
    return rms_tot, ms
