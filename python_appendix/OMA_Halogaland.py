import matplotlib.pyplot as plt
import numpy as np
import os
from nptdms import TdmsFile
import get_data as gd # This function works like readAcc, but extracts x-,y-,z-acceleration from an accelerometer pair
import OMA_functions as OMA
import time_synchronisation as ts
#%% Read corresponding file from all data loggers
path = os.getcwd()+'/../Data/Halogaland_2022_04_22'
anodes = ['/anode003', '/anode004', '/anode005', '/anode006', '/anode007', '/anode008', '/anode009', '/anode010'] # List of all data loggers
anode_names = ['A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
anemometers = ['W03-7-1', 'W04-15-1', 'W05-19-1', 'W07-28-1', 'W10-49-1'] # List of anemometer names for each data logger (the ones closest to accelerometers)

fileToRead = '2022-02-04-00-00-00Z.tdms'
files = []
for i in range(np.shape(anodes)[0]):
    files.append(TdmsFile.read(path + anodes[i] + '_' + fileToRead)) #tdms read is utilized here-> whole file are stored "in-memory", can be done more efficiently with using TdmsFile.open
    print('File nr. ' + str(i+1) + ' done.')
#%% Extracting chosen interval of prosessed data from all accelerometers

print(type(files[0]))

interval = 0
fs = 2
t_master = ts.time_master(files, ['W03', 'W04', 'W05', 'xx', 'W07', 'xx', 'xx', 'W10'])
# Finding shape to initialize matrices
_, acc_1y, _, _, _, _ = gd.readAcc('A03', files[0], 30, t_master, fs)
print(type(acc_1y))

# Saving accelerations from interval nr. interval from all accelerometers
acc_x_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
acc_y_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
acc_z_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
for i in range(np.shape(files)[0]):
    acc = gd.readAcc(anode_names[i], files[i], 30, t_master, fs)
    acc_x_all[:, i] = acc[0][interval, :]; acc_x_all[:, i+8] = acc[3][interval, :]
    acc_y_all[:, i] = acc[1][interval, :]; acc_y_all[:, i+8] = acc[4][interval, :] 
    acc_z_all[:, i] = acc[2][interval, :]; acc_z_all[:, i+8] = acc[5][interval, :]
   
acc_all = np.concatenate((acc_x_all, acc_y_all, acc_z_all), axis=1) # [1x values; 2x values; 1y values; 2y values; 1z values; 2z values]
#%% Cov-SSI - parameters
i = 24 # Maximum number of block rows
s = 6 # Stability level
orders = np.arange(2, 252, 2) # Array of what orders to include
stabcrit = {'freq': 0.05, 'damping': 0.1, 'mac': 0.1} # Default

#%% Stabilisation plot for chosen interval and parameters
f_n_sort, ksi_sort, phi_sort, fig, lambd_stab, phi_stab, orders_stab = OMA.modalParamOMA(acc_all, fs, orders, i, s, stabcrit=stabcrit, autoSpectrum=True)

#%% Clustering
import koma.clustering

pole_clusterer = koma.clustering.PoleClusterer(lambd_stab, phi_stab, orders_stab, min_cluster_size=10, min_samples=10, scaling={'mac':1.0, 'lambda_real':1.0, 'lambda_imag': 1.0})
prob_threshold = 0.5   #probability of pole to belong to cluster, based on estimated "probability" density function
args = pole_clusterer.postprocess(prob_threshold=prob_threshold, normalize_and_maxreal=True)

xi_auto, omega_n_auto, phi_auto, order_auto, probs_auto, ixs_auto = koma.clustering.group_clusters(*args)
xi_mean = np.array([np.mean(xi_i) for xi_i in xi_auto])
fn_mean = np.array([np.mean(om_i) for om_i in omega_n_auto])/2/np.pi

xi_std = np.array([np.std(xi_i) for xi_i in xi_auto])
fn_std = np.array([np.std(om_i) for om_i in omega_n_auto])/2/np.pi

# Print table
import pandas as pd
res_data = np.vstack([fn_mean, 100*xi_mean]).T
results = pd.DataFrame(res_data, columns=['$f_n$ [Hz]', r'$\xi$ [%]'])
print(results)

#%% Plots of mode shapes from OMA
print(f_n_sort.shape)
print(phi_sort.shape)

fig_shape = OMA.plotModeShape(phi_sort, 470)

plt.show()


