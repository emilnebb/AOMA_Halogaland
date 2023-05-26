import numpy as np
import os
from nptdms import TdmsFile
from python_appendix import get_data as gd # This function works like readAcc, but extracts x-,y-,z-acceleration from an accelerometer pair
from python_appendix import OMA_functions as OMA
from python_appendix import time_synchronisation as ts
from src.AOMA.plot import welch_plot

def main():

    #%% Read corresponding file from all data loggers
    path = os.getcwd()+'/../../../Data/Halogaland_2022_04_22'
    anodes = ['/anode003', '/anode004', '/anode005', '/anode006', '/anode007', '/anode008', '/anode009', '/anode010'] # List of all data loggers
    anode_names = ['A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
    anemometers = ['W03-7-1', 'W04-15-1', 'W05-19-1', 'W07-28-1', 'W10-49-1'] # List of anemometer names for each data logger (the ones closest to accelerometers)

    fileToRead = '2022-02-04-00-00-00Z.tdms'
    files = []
    for i in range(np.shape(anodes)[0]):
        files.append(TdmsFile.read(path + anodes[i] + '_' + fileToRead)) #tdms read is utilized here-> whole file are stored "in-memory", can be done more efficiently with using TdmsFile.open
        print('File nr. ' + str(i+1) + ' done.')
    #%% Extracting chosen interval of prosessed data from all accelerometers


    interval = 0
    fs = 2
    t_master = ts.time_master(files, ['W03', 'W04', 'W05', 'xx', 'W07', 'xx', 'xx', 'W10'])
    # Finding shape to initialize matrices
    _, acc_1y, _, _, _, _ = gd.readAcc('A03', files[0], 30, t_master, fs)

    # Saving accelerations from interval nr. interval from all accelerometers
    acc_x_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
    acc_y_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
    acc_z_all = np.zeros([np.shape(acc_1y)[1], np.shape(files)[0]*2])
    print(acc_y_all.shape)
    for i in range(np.shape(files)[0]):
        acc = gd.readAcc(anode_names[i], files[i], 30, t_master, fs)
        acc_x_all[:, i] = acc[0][interval, :]; acc_x_all[:, i+8] = acc[3][interval, :]
        acc_y_all[:, i] = acc[1][interval, :]; acc_y_all[:, i+8] = acc[4][interval, :]
        acc_z_all[:, i] = acc[2][interval, :]; acc_z_all[:, i+8] = acc[5][interval, :]

    acc_all = np.concatenate((acc_x_all, acc_y_all, acc_z_all), axis=1) # [1x values; 2x values; 1y values; 2y values; 1z values; 2z values]

    print("Shape of vector going into cov-SSI (with acceleration data in x,y and z direction)= " + str(acc_all.shape))
    welch_plot(acc_all[:, 20], 2, 5)

    #%% Cov-SSI - parameters
    i = 24 # Maximum number of block rows
    s = 6 # Stability level
    orders = np.arange(2, 252, 2) # Array of what orders to include
    stabcrit = {'freq': 0.05, 'damping': 0.1, 'mac': 0.1} # Default
    print(orders)
    #%% Stabilisation plot for chosen interval and parameters
    f_n_sort, ksi_sort, phi_sort, fig, lambd_stab, phi_stab, orders_stab, lambd, phi = OMA.modalParamOMA(acc_all, fs, orders, i, s, stabcrit=stabcrit, autoSpectrum=True)



    return f_n_sort, ksi_sort, phi_sort, fig, lambd_stab, phi_stab, orders_stab, acc_all, lambd, phi



