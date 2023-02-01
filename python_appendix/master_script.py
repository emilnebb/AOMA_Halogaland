#%% Import necessary modules and packages
import numpy as np
from nptdms import TdmsFile
import os
import time_synchronisation as ts
import get_data as gd
import fitting as fit
#%% Necessary paths
path = 'H:'
nodes_folder = ['\\anode003', '\\anode004', '\\anode005', '\\anode006', '\\anode007', '\\anode008', '\\anode009', '\\anode010']
save_path = os.getcwd() + '\Results'
#%% INFO
acc_nodes = ['A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
wind_nodes = ['W03', 'W04', 'W05', 'xx', 'W07', 'xx', 'xx', 'W10']
anemometers = ['W03-7-1', 'W04-15-1', 'W05-17-1', 'W05-18-1', 'W05-19-1', 'W05-19-2', 'W07-28-1', 'W10-45-1', 'W10-47-1', 'W10-49-1']
acc_datasets =  ['-1x', '-1y', '-1z', '-2x', '-2y', '-2z']
Fs = 2 # The final frequency of the data
#%% Extracting data from midspan   
folder = path + nodes_folder[4]  # anode007
acc_node = acc_nodes[4]          # W07
anemometer = anemometers[6]      # W07-28-1

filenames = []
for file in os.listdir(folder):
    if (file.endswith('.tdms') and file.startswith('2022')):
        filenames.append(file)

for i in range(len(filenames)):
    try:      
        tdms_file = TdmsFile.read(folder+'\\'+filenames[i])
    except:
         print('Error opening the file ' + filenames[i])
         continue
    else:
        print('File number: ' + str(i+1) + ' read')        
        #Computing master time vector
        t_master = ts.time_master([tdms_file], [wind_nodes[4]])        
        #Extracting data
        V, meandir, u, w, Iu, Iw, sigma_y, sigma_z, sigma_theta = gd.getCharacteristics(tdms_file, acc_node, anemometer, t_master, Fs)
        print('Finished extracting data from file ' + str(i+1))
        # Deleting intervals with mean wind speed below 3 m/s
        ind = np.sort(np.argwhere(V < 3))
        V = np.delete(V, ind); meandir = np.delete(meandir, ind); u = np.delete(u, ind, axis=1); w = np.delete(w, ind, axis=1)
        Iu = np.delete(Iu, ind); Iw = np.delete(Iw, ind); sigma_y = np.delete(sigma_y, ind); sigma_z = np.delete(sigma_z, ind); sigma_theta = np.delete(sigma_theta, ind)
        # Obtaining the spectral values Au and Aw for the intervals
        Au = np.zeros(len(V))
        Aw = np.zeros(len(V))
        lse_Au = np.zeros(len(V))
        lse_Aw = np.zeros(len(V))    
        for k in range(len(V)):
            print('Calculating A for interval nr. ' + str(k+1))
            Au[k], Aw[k], lse_Au[k], lse_Aw[k] = fit.autoSpectraParam(Fs, u[:,k], w[:,k], V[k], True)   
        tdms_file.close()
        # Saving the extracted data
        variables = [V, meandir, Iu, Iw, sigma_y, sigma_z, sigma_theta, Au, Aw, lse_Au, lse_Aw]
        file_data = np.zeros((len(V), len(variables)))
        for k in range(np.shape(file_data)[1]):
            file_data[:,k] = variables[k]            
        turbulence = [u, w]
        turb_data = np.zeros((np.shape(u)[0], np.shape(u)[1], 2))
        for k in range(np.shape(turb_data)[2]):
            turb_data[:,:, k] = turbulence[k]
        
        np.savetxt(save_path + nodes_folder[4] + '\\' + filenames[i][0:20] + '.csv', file_data, delimiter=',')
        np.savetxt(save_path + nodes_folder[4] + '\\' + 'Turbulence' + '\\' + filenames[i][0:20] + '_u.csv', turb_data[:,:,0], delimiter=',')
        np.savetxt(save_path + nodes_folder[4] + '\\' + 'Turbulence' + '\\' + filenames[i][0:20] + '_w.csv', turb_data[:,:,1], delimiter=',')
#%% Extract K_u,w from fitting of data from anemometers W05-19-1, W05-18-1, W05-17-1 and W04-15-1
dx = np.array([20, 40, 80])         # Distances between anemometers

folder1 = path + nodes_folder[1]
fileNames_A04 = []
for file in os.listdir(folder1):
    if (file.endswith('.tdms') and file.startswith('2022')):
        fileNames_A04.append(file)

folder2 = path + nodes_folder[2]
fileNames_A05 = []
for file in os.listdir(folder2):
    if (file.endswith('.tdms') and file.startswith('2022')):
        fileNames_A05.append(file)

fileNames_inBoth = list(set(fileNames_A04).intersection(fileNames_A05))
fileNames_inBoth.sort()

for file in fileNames_inBoth: # Loop through all files
    print('Reading file ' + file)
    try:     
        cur_file_A04 = TdmsFile.read(path + nodes_folder[1] + '\\' + file)
        cur_file_A05 = TdmsFile.read(path + nodes_folder[2] + '\\' + file)
    except:
         print('Error opening the file ' + file)
         continue
    else: 
        #Computing master timevector
        t_master = ts.time_master([cur_file_A04, cur_file_A05], ['W04', 'W05'])
        # Extracting data from anemometers
        u19, w19, V19, meandir19 = gd.readWind('W05-19-1', cur_file_A05, 10, t_master, Fs)
        u18, w18, V18, meandir18 = gd.readWind('W05-18-1', cur_file_A05, 10, t_master, Fs)
        u17, w17, V17, meandir17 = gd.readWind('W05-17-1', cur_file_A05, 10, t_master, Fs)
        u15, w15, V15, meandir15 = gd.readWind('W04-15-1', cur_file_A04, 10, t_master, Fs)
        # Deleting intervals that have mean direction deviating more than 20 degrees from 90 and 270 and/or mean wind speed below 3 m/s
        ind = []
        dum1 = np.argwhere(np.logical_or((np.logical_or(meandir19 < 70, meandir19 > 290)), np.logical_and(meandir19 < 250, meandir19 > 110)))
        dum2 = np.argwhere(np.logical_or((np.logical_or(meandir18 < 70, meandir18 > 290)), np.logical_and(meandir18 < 250, meandir18 > 110)))
        dum3 = np.argwhere(np.logical_or((np.logical_or(meandir17 < 70, meandir17 > 290)), np.logical_and(meandir17 < 250, meandir17 > 110)))
        dum4 = np.argwhere(np.logical_or((np.logical_or(meandir15 < 70, meandir15 > 290)), np.logical_and(meandir15 < 250, meandir15 > 110)))
        dum5 = np.argwhere(np.logical_or(np.logical_or(np.logical_or(V19 < 3, V18 <3), V17 <3), V15 < 3))
        ind = np.concatenate((np.concatenate((np.concatenate((np.concatenate((dum1, dum2)), dum3)), dum4)), dum5))        
        ind = np.sort((list(set(ind.flatten()))))    
        if len(ind)!=0:
            u19 = np.delete(u19, ind, axis=1); V19 =  np.delete(V19, ind); u18 = np.delete(u18, ind, axis=1); V18 = np.delete(V18, ind); u17 = np.delete(u17, ind, axis=1); V17 = np.delete(V17, ind); u15 = np.delete(u15, ind, axis=1); V15 = np.delete(V15, ind) 
            w19 = np.delete(w19, ind, axis=1); w18 = np.delete(w18, ind, axis=1); w17 = np.delete(w17, ind, axis=1); w15 = np.delete(w15, ind, axis=1)
        # Obtaining Ku and Kw for intervals
        Ku = np.zeros(np.shape(u19)[1]); lse_Ku = np.zeros(np.shape(u19)[1])
        Kw = np.zeros(np.shape(u19)[1]); lse_Kw = np.zeros(np.shape(u19)[1])
        for interval in range(np.shape(u19)[1]):            
            print('Calculating K for interval nr. ' + str(interval+1))         
            u = np.array([u19[:, interval] + V19[interval], 
                          u18[:, interval] + V18[interval],
                          u17[:, interval] + V17[interval],
                          u15[:, interval] + V15[interval]]).T           
            w = np.array([w19[:, interval], w18[:, interval], w17[:, interval], w15[:, interval]]).T           
            Ku[interval], Kw[interval], lse_Ku[interval], lse_Kw[interval] = fit.normCrossSpectraParam(Fs, u, w, dx)            
        cur_file_A04.close()
        cur_file_A05.close()
        # Saving the extracted values
        variables = [Ku, Kw, lse_Ku, lse_Kw, np.mean(np.array([V19, V18, V17, V15]), axis=0)]
        file_data = np.zeros((len(Ku), len(variables)))
        for k in range(np.shape(file_data)[1]):
            file_data[:,k] = variables[k]
        np.savetxt(save_path  + '\\K\\' + file[0:20] + '.csv', file_data, delimiter=',')
