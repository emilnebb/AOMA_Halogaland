import numpy as np
import buffeting_response as br
import os
#%% Predicted buffeting response using semi-probabilistic approach
I_u_all = [0.0726, 0.0741, 0.0735]
I_w_all = [0.0456, 0.0408, 0.0346]
Au = 8.3
Aw = 1.9
Ku = 8.2
Kw = 8.6
# RUN the calculation  
U = np.linspace(0.1, 25, 30) # Vector of mean wind velocities
U = np.concatenate((U, np.array[9.95, 10.05, 14.95, 15.05]))

file = 'Case 1'

for i in range(np.shape(U)[0]):
    if len(I_u_all) > 1:
        if U[i] < 10:
            I_u = I_u_all[0]
            I_w = I_w_all[0]
        elif U[i] >= 10 and U[i] < 15:
            I_u = I_u_all[1]
            I_w = I_w_all[1]    
        elif U[i] >= 15:
            I_u = I_u_all[2]
            I_w = I_w_all[2]  
    print('V = '+str(U[i]))
    rms_tot, ms = br.RMS(U[i], I_u, I_w, Au, Aw, Ku, Kw, 140, spectra_type='kaimal', filename=file)
#%% Predicted buffeting response using probabilistic approach
# Retrieving the simulated values of sigma_u and sigma_w
simulated = os.getcwd() + '\Results' + '\\Predicted\Simulations'
V_east = []; std_u_east = []; std_w_east = [];
V_west = []; std_u_west = []; std_w_west = [];
for file in os.listdir(simulated):
    if (file.endswith('east.csv')):
        res = np.loadtxt(simulated + '\\' + file, delimiter=',')
        for i in range(np.shape(res)[0]):
            V_east.append(res[i][0])
            std_u_east.append(res[i][1])
            std_w_east.append(res[i][2])
    if (file.endswith('west.csv')):
        res = np.loadtxt(simulated + '\\' + file, delimiter=',')
        for i in range(np.shape(res)[0]):
            V_west.append(res[i][0])
            std_u_west.append(res[i][1])
            std_w_west.append(res[i][2])
           
# Predicted response using probabilistic approach for easterly winds
Au = 13.9
Aw = 1.7
Ku = 7.8
Kw = 8.9
file = 'sim_east'
for i in range(0, len(V_east)):
    I_u = std_u_east[i]/V_east[i]
    I_w = std_w_east[i]/V_east[i]    
    print('   Calculating for V_east nr. ' + str(i+1) + ' out of ' + str(np.shape(V_east)[0]))
    rms_tot, ms = br.RMS(V_east[i], I_u, I_w, Au, Aw, Ku, Kw, 140, spectra_type='kaimal', filename=file)   

# Predicted response using probabilistic approach for westerly winds
Au = 9.4
Aw = 1.8
Ku = 7.8
Kw = 8.9
file = 'sim_west'
for i in range(np.shape(V_west)[0]):
    I_u = std_u_west[i]/V_west[i]
    I_w = std_w_west[i]/V_west[i]    
    print('   Calculating for V_west nr. ' + str(i+1) + ' out of ' + str(np.shape(V_west)[0]))
    rms_tot, ms = br.RMS(V_west[i], I_u, I_w, Au, Aw, Ku, Kw, 140, spectra_type='kaimal', filename=file)
    