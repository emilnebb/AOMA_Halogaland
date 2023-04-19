# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:19:19 2023

@author: oyvinpet
"""

#%%

import h5py    
import numpy as np    
import matplotlib.pyplot as plt

hf = h5py.File('HalogalandModel_exportmodal.h5', 'r')

# Frequencies
f = np.array(hf.get('f'))

# Generalized mass
gm = np.array(hf.get('gm'))

# Node coordinates all nodes (size [N_nodes,4])
nodecoord = np.array(hf.get('nodecoord'))

# Mode shape matrix for all DOFs (size [6*N_nodes,N_modes])
phi = np.array(hf.get('phi'))

# Labels corresponding to each row of phi (each DOF), as a list of strings
phi_label_temp = np.array(hf.get('phi_label'))
phi_label=phi_label_temp[:].astype('U10').ravel().tolist()

# The DOFs are as follows:
# U1=x=longitudinal translation (along bridge)
# U2=y=horizontal translation
# U3=z=vertical translation
# UR1=rotation about x aka. torsion
# UR2=rotation about y aka. vertical bending
# UR3=rotation about z aka. horizontal bending


#%% Get node coordinates and mode shape for bridge deck only

# Nodes in bridge deck (total 573)
node_deck=np.arange(1004,1576+1,1)

# Create list index of nodes in bridge deck
index_node_deck=[]

for k in np.arange(len(node_deck)):
    index_node_deck.append(np.argwhere(node_deck[k]==nodecoord[:,0])[0,0])
   
nodecoord_deck=nodecoord[index_node_deck,:]

   
# Create list of index of y-DOFs,z-DOFs, and t-DOFs in bridge deck
index_y=[]
index_z=[]
index_t=[]

for k in np.arange(len(node_deck)):
    str_y=str(node_deck[k]) + '_U2'
    index_y.append(phi_label.index(str_y))  
    
    str_z=str(node_deck[k]) + '_U3'
    index_z.append(phi_label.index(str_z))  
    
    str_t=str(node_deck[k]) + '_UR1'
    index_t.append(phi_label.index(str_t))
    
    
phi_y=phi[index_y,:]
phi_z=phi[index_z,:]
phi_t=phi[index_t,:]


#%% Plot single mode

mode_plot=12 # Mode number to plot
scale_factor=1e4 # Scale factor

plt.figure()

x=nodecoord_deck[:,1] #x-coordinate of deck nodes

h1=plt.plot(x,phi_y[:,mode_plot]*scale_factor, label='Horizontal')
h2=plt.plot(x,phi_z[:,mode_plot]*scale_factor, label='Vertical')
h3=plt.plot(x,phi_t[:,mode_plot]*scale_factor, label='Torsion')

plt.xlabel('x [m]')
plt.ylabel('Modal deflection [m or rad]')

plt.legend()

plt.show()



