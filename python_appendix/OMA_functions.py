###### Import modules
import numpy as np
import koma.oma as oma
import koma.modal as modal
from matplotlib import pyplot as plt
import scipy.signal as sci

def modalParamOMA(data, fs, orders, i, s, stabcrit={'freq': 0.05, 'damping': 0.1, 'mac': 0.1}, autoSpectrum=False):   
    """
    Arguments:
    data         : data matrix, each column containing data from one accelerometer 
    fs           : sampling frequency
    orders       : array of what orders to perform the Cov-SSI for
    i            : maximum number of block rows
    s            : stability level
    stabcrit     : stabilization criteria
    autospectrum : parameter to decide whether or not to plot an auto-spectrum in the stabilization plot   
    Returns:
    f_n_sort, ksi_sort, phi_sort : natural frequencies, damping ratios and mode shapes from stable poles, sorted based on the values of the 
                                   natural frequencies
    fig                          : stabilization plot
    -----------------------------------
    Function for computing Cov-SSI to obtain modal parameters from stable poles and stabilization plot
    """   
    ## Find all complex poles and eigenvectors for all orders
    lambd, phi = oma.covssi(data=data, fs=fs, i=i, orders=orders, weighting='none', matrix_type='hankel', algorithm='shift', showinfo=True, balance=True)    
    ## Find stable poles from all poles
    lambd_stab, phi_stab, orders_stab, idx_stab = oma.find_stable_poles(lambd, phi, orders, s, stabcrit=stabcrit, valid_range={'freq': [0, np.inf], 'damping':[0, np.inf]}, indicator='freq', return_both_conjugates=False)   
    ## Natural frequencies, damping ratios and mode shapes for stable poles, sorted after the value of the natural frequencies
    w_n = np.abs(lambd_stab)
    f_n = w_n / (2*np.pi)
    ksi = - np.real(lambd_stab) / np.abs(lambd_stab) 
    
    f_n_sort_index = np.argsort(f_n)
    
    f_n_sort = f_n[f_n_sort_index]
    ksi_sort = ksi[f_n_sort_index]
    phi_sort = phi_stab[:, f_n_sort_index]
    f_size = 20
    ## Stabilization plot
    if autoSpectrum == False:
        fig = plt.figure(figsize=(20,4))
        plt.plot(f_n, orders_stab, 'o', color='black')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('System order, n')
        plt.ylim(0, max(orders))
        plt.xlim(0,1)
    else:   
        fig, axs = plt.subplots(4,1, figsize=(20,14))
        
        ## Auto-spectum of sensor pair in the midspan
        Nwelch = 3
        Nwindow = np.floor(np.max(np.shape(data))/Nwelch) # Length of window
        
        B = 18.6
        # Transforming to find y, z, theta component for the sensor pair at the midspan
        y = (data[:,20] + data[:,28])/2 
        z = (data[:,36] + data[:,44])/2 
        theta = (-data[:,36] + data[:,44])/B

        # Spectra        
        f, Sy = sci.csd(y, y, fs, nperseg = Nwindow)
        f, Sz = sci.csd(z, z, fs, nperseg = Nwindow)
        f, Stheta = sci.csd(theta, theta, fs, nperseg = Nwindow)
        
        plt.sca(axs[0])
        plt.plot(f_n, orders_stab, 'o', color='black')
        plt.xlim(0,1)
        plt.ylim(0, max(orders))
        plt.xlabel('(a)', fontsize=f_size)
        plt.ylabel('System order, n', fontsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        
        plt.sca(axs[1])
        plt.plot(f, Sy, color='blue')
        plt.xlim(0,1)
        plt.xlabel('(b)', fontsize=f_size)
        plt.ylabel(r'$S_{\ddot{y} \ddot{y}}$  [$m^2/s^3$]', fontsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        plt.ticklabel_format(axis="y", style='sci', scilimits=(-1,-1))
        
        plt.sca(axs[2])
        plt.plot(f, Sz, color='purple')
        plt.xlim(0,1)
        plt.xlabel('(c)', fontsize=f_size)
        plt.ylabel(r'$S_{\ddot{z} \ddot{z}}$  [$m^2/s^3$]', fontsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        plt.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        
        plt.sca(axs[3])
        plt.plot(f, Stheta, color='red')
        plt.xlim(0,1)
        plt.xlabel('f [Hz] \n \n (d)', fontsize=f_size)
        plt.ylabel(r'$S_{\ddot{\theta} \ddot{\theta}}$  [$rad^2/s^3$]', fontsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        plt.ticklabel_format(axis="y", style='sci', scilimits=(-3,-3))
        
        plt.tight_layout()    
    
    return f_n_sort, ksi_sort, phi_sort, fig, lambd_stab, phi_stab, orders_stab
    
       
def plotModeShape(phi, i_phi_plot):

    """
    Arguments:
    phi        : mode shape matrix, each column containing mode shape values for x1, x2, y1, y2, z1, z2 at the sensor locations
    i_phi_plot : index of which mode shape to plot   
    Returns:
    fig        : plot of mode shapes
    ---------------------------------
    Function plots mode shapes obtained from Cov-SSI
    """     
    B = 18.6 # Width of bridge girder
    
    phi_x = modal.maxreal((phi[0:8,:]+phi[8:16,:])/2)
    phi_y = modal.maxreal((phi[16:24,:]+phi[24:32,:])/2)
    phi_z = modal.maxreal((phi[32:40,:]+phi[40:48,:])/2)
    phi_theta = modal.maxreal((-phi[32:40,:]+phi[40:48,:])/B)
    
    x_sensors = np.array([-572.5, -420, -300, -180, -100, 0, 100, 260, 420, 572.5]) # Sensor x-coordinates - [TOWER, A03, A04, A05, A06, A07, A08, TOWER]
    # Calculating common y-lim for all plots
    ylim = 1.09*max([max(abs(phi_x[:, i_phi_plot])), max(abs(phi_y[:, i_phi_plot])), max(abs(phi_z[:, i_phi_plot])), max(abs(phi_theta[:, i_phi_plot]))])
    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(7,10))   
    axs[0].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_x[:, i_phi_plot]), np.array([0]))), color='black')
    axs[0].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[0].axvline(0, color='grey', linestyle=':', linewidth=1)  
    axs[0].set_title('Longitudinal mode shape')
    axs[0].set_ylim(-ylim, ylim)
    
    axs[1].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_y[:, i_phi_plot]), np.array([0]))), color='black')
    axs[1].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[1].axvline(0, color='grey', linestyle=':', linewidth=1)  
    axs[1].set_title('Horizontal mode shape')
    axs[1].set_ylim(-ylim, ylim)
    
    axs[2].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_z[:, i_phi_plot]), np.array([0]))), color='black')
    axs[2].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[2].axvline(0, color='grey', linestyle=':', linewidth=1)  
    axs[2].set_title('Vertical mode shape')
    axs[2].set_ylim(-ylim, ylim)
    
    axs[3].plot(x_sensors, np.concatenate((np.array([0]), np.real(phi_theta[:, i_phi_plot]), np.array([0]))), color='black') 
    axs[3].axhline(0, color='grey', linestyle=':', linewidth=1)
    axs[3].axvline(0, color='grey', linestyle=':', linewidth=1)  
    axs[3].set_title('Torsional mode shape')
    axs[3].set_ylim(-ylim, ylim)
    
    plt.tight_layout() 
    
    return fig
      
def getModeShapeInSpecifiedDirection(phi, i_phi_plot, direction):

    """
    Arguments:
    phi        : mode shape matrix, each column containing mode shape values for x1, x2, y1, y2, z1, z2 at the sensor locations
    i_phi_plot : index of which mode shape to plot
    direction  : type of mode shape : 'x' / 'y' / 'z' / 'theta'   
    Returns:
    phi        : normalized mode shape values in specified direction
    -------------------------------------
    Function plots mode shape in specified direction
    """       
    B = 18.6
    
    if direction == 'x':
        norm, _ = modal.normalize_phi(np.real(modal.maxreal((phi[0:8,:]+phi[8:16,:])/2)))
        phi = norm[:, i_phi_plot]
    elif direction == 'y':
        norm, _ = modal.normalize_phi(np.real(modal.maxreal((phi[16:24,:]+phi[24:32,:])/2)))
        phi = norm[:, i_phi_plot]
    elif direction == 'z':
        norm, _ = modal.normalize_phi(np.real(modal.maxreal((phi[32:40,:]+phi[40:48,:])/2)))
        phi = norm[:, i_phi_plot]
    else:
        norm, _ = modal.normalize_phi(np.real(modal.maxreal((-phi[32:40,:]+phi[40:48,:])/B)))
        phi = norm[:, i_phi_plot]
    
    
    return phi

    