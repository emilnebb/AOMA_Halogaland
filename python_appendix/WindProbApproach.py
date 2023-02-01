#%% Modules
import numpy as np
import os
import logNormalDist as lognorm

#%% Info for loading data
nodes_folder = ['\\anode003', '\\anode004', '\\anode005', '\\anode006', '\\anode007', '\\anode008', '\\anode009', '\\anode010']
save_path = os.getcwd() + '\Results'
#%% Retrieve data from csv-files
data = []
files = []
for file in os.listdir(save_path + nodes_folder[4]):
    if (file.endswith('.csv') and os.path.getsize(save_path + nodes_folder[4] + '\\' + file) > 0):
        files.append(file)
        data.append(np.loadtxt(save_path + nodes_folder[4] + '\\' + file, delimiter=','))     
elem = 0
for i in range(len(data)):
    if data[i].ndim == 1:
        elem += 1
    else:
        elem += np.shape(data[i])[0]
#%% Array of variables
V = np.zeros(elem)
meandir = np.zeros(elem)
Iu = np.zeros(elem)
Iw = np.zeros(elem)
Au = np.zeros(elem)
Aw = np.zeros(elem)
c = 0
for i in range(len(data)):
    if data[i].ndim == 1:
        V[c] = data[i][0]
        meandir[c] = data[i][1]
        Iu[c] = data[i][2]
        Iw[c] = data[i][3] 
        Au[c] = data[i][7]
        Aw[c] = data[i][8] 
        c += 1
    else:
        for k in range(np.shape(data[i])[0]):
            V[c] = data[i][k,0]
            meandir[c] = data[i][k,1]
            Iu[c] = data[i][k,2]
            Iw[c] = data[i][k,3] 
            Au[c] = data[i][k,7]
            Aw[c] = data[i][k,8] 
            c += 1
sigma_u = Iu * V
sigma_w = Iw * V
#%% Dividing data into easterly and westerly wind
index_east = np.argwhere(np.logical_and(meandir > 0, meandir <= 180))
index_west = np.argwhere(np.logical_and(meandir > 180, meandir <= 360))

V_east = V[index_east]
V_west = V[index_west]
sigma_u_east = sigma_u[index_east]
sigma_u_west = sigma_u[index_west]
sigma_w_east = sigma_w[index_east]
sigma_w_west = sigma_w[index_west]
#%% Correlation coefficients matrix - only wind above 11
corr_east = np.corrcoef(np.array([sigma_u_east[np.argwhere(V_east[:,0] >= 11)[:,0],0], sigma_w_east[np.argwhere(V_east[:,0] >= 11)[:,0],0]]))
corr_west = np.corrcoef(np.array([sigma_u_west[np.argwhere(V_west[:,0] >= 11)[:,0],0], sigma_w_west[np.argwhere(V_west[:,0] >= 11)[:,0],0]]))
#%% Conditional probability parameters and correlation coefficients
east_sort = np.argsort(V_east[:,0])
V_e = V_east[east_sort,0]
sigma_u_e = sigma_u_east[east_sort,0]
sigma_w_e = sigma_w_east[east_sort,0]

west_sort = np.argsort(V_west[:,0])
V_w = V_west[west_sort,0]
sigma_u_w = sigma_u_west[west_sort,0]
sigma_w_w = sigma_w_west[west_sort,0]

## Easterly wind
V_east_list = []
sigma_u_east_list = []
sigma_w_east_list = []
start = 0
end = 200
move = end
num = np.floor(np.shape(V_e)[0]/end-1)
for i in np.arange(int(num)):
    V_east_list.append(V_e[start:end])
    sigma_u_east_list.append(sigma_u_e[start:end])
    sigma_w_east_list.append(sigma_w_e[start:end])  
    start = end
    end = end + move
V_east_list.append(V_e[start:])
sigma_u_east_list.append(sigma_u_e[start:])
sigma_w_east_list.append(sigma_w_e[start:])
V_east_mean = np.zeros([len(V_east_list)])
param_sigma_u_east = np.zeros([len(sigma_u_east_list), 2])
param_sigma_w_east = np.zeros([len(sigma_w_east_list), 2])
corr_east_list = np.zeros([len(V_east_list)])
for i in np.arange(len(sigma_u_east_list)):
    param_sigma_u_east[i,:] = lognorm.logNormParam(sigma_u_east_list[i])
    param_sigma_w_east[i,:] = lognorm.logNormParam(sigma_w_east_list[i])
    V_east_mean[i] = np.mean(V_east_list[i])
    corr_east_list[i] = np.corrcoef(np.array([sigma_u_east_list[i], sigma_w_east_list[i]]))[1,0]
    
## Westerly wind
V_west_list = []
sigma_u_west_list = []
sigma_w_west_list = []
start = 0
end = 150
move = end
num = np.floor(np.shape(V_w)[0]/end-1)
for i in np.arange(int(num)):
    V_west_list.append(V_w[start:end])
    sigma_u_west_list.append(sigma_u_w[start:end])
    sigma_w_west_list.append(sigma_w_w[start:end])
    start = end
    end = end + move 
V_west_list.append(V_w[start:])
sigma_u_west_list.append(sigma_u_w[start:])
sigma_w_west_list.append(sigma_w_w[start:])
V_west_mean = np.zeros([len(V_west_list)])
param_sigma_u_west = np.zeros([len(sigma_u_west_list), 2])
param_sigma_w_west = np.zeros([len(sigma_w_west_list), 2])
corr_west_list = np.zeros([len(V_west_list)])
for i in np.arange(len(sigma_u_west_list)):
    param_sigma_u_west[i,:] = lognorm.logNormParam(sigma_u_west_list[i])
    param_sigma_w_west[i,:] = lognorm.logNormParam(sigma_w_west_list[i])
    V_west_mean[i] = np.mean(V_west_list[i])
    corr_west_list[i] = np.corrcoef(np.array([sigma_u_west_list[i], sigma_w_west_list[i]]))[1,0]

