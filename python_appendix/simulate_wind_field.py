import os
import numpy as np
def simulate_wind(V, direction):
    '''
    Arguments:
    V           : mean wind velocity of a 10 min recording
    direction   : the direction of the mean wind, 'east' or 'west' 
    Returns:
    sigma : sample of standard deviation of along wind and vertical turbulence
    --------------------------------------------------------------------
    Functions takes in a mean wind velocity, and establishes the conditional distributions of the wind field model. First, a set of multivariate correlated normally distributed random variables are generated, and then the parameters are obtained by taking the natural exponent 
    '''     
    if direction=='east':
        my_t = [0.07046*V - 0.8886, 0.04191*V - 1.1549]
        sigma_t = [0.3211, 0.2481] 
        rho = np.array([[1, 0.8713], [0.8713, 1]])  
    else:
        my_t = [0.04458*V - 0.6850, 0.02141*V - 0.9824]
        sigma_t = [0.4324, 0.4604] 
        rho = np.array([[1, 0.9268], [0.9268, 1]])
    cov = np.zeros((len(sigma_t), len(sigma_t)))
    for i in range(len(sigma_t)):
        for j in range(len(sigma_t)):
            cov[i,j] = np.log(rho[i,j]*np.sqrt(np.exp(sigma_t[i]**2)-1)*np.sqrt(np.exp(sigma_t[j]**2) - 1) + 1)
    # Generating a sample
    rng = np.random.default_rng()
    x = rng.multivariate_normal(my_t, cov)
    sigma = np.exp(x)
    
    return sigma

#%% Import measured data
nodes_folder = ['\\anode003', '\\anode004', '\\anode005', '\\anode006', '\\anode007', '\\anode008', '\\anode009', '\\anode010']

save_path = os.getcwd() + '\Results'

# Retrieve [V, meandir, Iu, Iw, sigma_y, sigma_z, sigma_theta, Au, Aw, lse_Au, lse_Aw] from txt-files

# variables = [V, meandir, Iu, Iw, sigma_y, sigma_z, sigma_theta, Au, Aw, lse_Au, lse_Aw]
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
#Array of all V and meandir
V = np.zeros(elem)
meandir = np.zeros(elem)
c = 0
for i in range(len(data)):
    if data[i].ndim == 1:
        V[c] = data[i][0]
        meandir[c] = data[i][1]
        c += 1
    else:
        for k in range(np.shape(data[i])[0]):
            V[c] = data[i][k,0]
            meandir[c] = data[i][k,1]
            c += 1
#%% ------ SIMULATION -----------------
V_lim = 11 # Lowest mean wind included
V_upper = 30 # Highest mean wind included
V_east = V[np.argwhere(np.logical_and(np.logical_and(V >=V_lim, V <= V_upper), np.logical_and(meandir > 0, meandir <= 180)))]
V_west = V[np.argwhere(np.logical_and(np.logical_and(V >=V_lim, V <= V_upper), np.logical_and(meandir > 180, meandir <= 360)))]
sigma_east = np.zeros((len(V_east), 2))
sigma_west = np.zeros((len(V_west), 2))
for i in range(len(V_east)):
    sigma_east[i,:] = simulate_wind(V_east[i][0], 'east')
for i in range(len(V_west)):
    sigma_west[i,:] = simulate_wind(V_west[i][0], 'west')   
