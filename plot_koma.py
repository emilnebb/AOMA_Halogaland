### Wrtten by Ole Bjørn Falang ###

import numpy as np
import scipy.linalg as sp
import scipy.signal
import matplotlib.pyplot as plt
import random


def plot_stab_from_KOMA(sorted_freqs,sorted_orders,*args,lambd_stab=None,stab_orders=np.zeros(1),lambd=None,all_orders=np.zeros(1),true_w=None,xmin=0,xmax=None,**kwargs):

    """Returns a figure with the clusters color coded  
    and with lines in between the poles in one cluster
    input: 
    Arguments: 
        sorted_freqs: 2darray
            first axis correspons to each order, second is freqs
            in that order
        sorted_orders: 1d array
            orders tht corresponds to sorted_freqs
        *args: 
            lambd: 2darray
                same type as sorted_freqs. Eigenvalues or 
                frequencies 
            all_orders: 1d array
                orders that correspons to lambd.
            true_w: 1darray
                Array ot the analythical natural frequencies
            figsize: (int,int)
            plot_negatives: bool
    
    """

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    if not xmax: 
        xmax=np.amax(sorted_freqs[-1])*1.2
    if not xmin: 
        xmin=np.amin(sorted_freqs[0])*0.8
    

    # Make an arrays of colors. 
    colors=[]
    num=len(sorted_freqs)+8
    root=int(num**(1/3))
    root_rest=int(num/root**2)+1
    colors=[]
    for r in range(root):
        for g in range(root):
            for b in range(root_rest): 
                colors.append([r/root,g/root,b/root_rest])
    random.shuffle(colors)
           
    #create figure and plot scatter
    if 'figsize' in kwargs.keys():
        fig_s=kwargs['figsize']
    else:
        fig_s=(9,9)
    figur_h=plt.figure(figsize=fig_s)
    axes=figur_h.add_subplot(1,1,1)
    
    s_dot=fig_s[1]*3

    #The analythical freqs: 
    if np.any(true_w):
        lab=0
        for w in true_w:
            if lab==0:
                axes.axvline(w,linestyle='--',label='Analytichal freqs')
            else: 
                axes.axvline(w,linestyle='--')
            lab=1

    # plotting all the discarded poles: 
    if len(all_orders)>1:
        for i,order_i in enumerate(all_orders):
            lambd_i=lambd[i]         
            # xi=-np.real(lambd_i)/np.abs(lambd_i)
            # freq_i=np.imag(lambd_i)/(np.sqrt(1-xi**2))
            freq_i=np.abs(lambd_i)**0.5
            order_is=np.ones_like(freq_i)*order_i
            col='black'
            if i==0:
                axes.scatter(np.real(freq_i),order_is,color=col,
                marker='x',s=s_dot,label='Deleted poles')
            else:
                axes.scatter(np.real(freq_i),order_is,
                color=col,marker='x',s=s_dot)

    #plotting the poles that were deemed stable: 
    if len(stab_orders)>1:
        for i,order_i in enumerate(stab_orders):
            lambd_i=lambd_stab[i]         
            # xi=-np.real(lambd_i)/np.abs(lambd_i)
            # freq_i=np.imag(lambd_i)/(np.sqrt(1-xi**2))
            freq_i=np.abs(lambd_i)
            order_is=np.ones_like(freq_i)*order_i
            col='red'
            if i==0:
                axes.scatter(np.real(freq_i**0.5),order_is,color=col,
                marker='o',s=s_dot,label='Stable poles')
            else:
                axes.scatter(np.real(freq_i**0.5),order_is,
                color=col,marker='o',s=s_dot*1.5)

    
    
    #plotting the poles to keep, and the lines between them
    for i,order_i in enumerate(sorted_orders):
        freq_i=sorted_freqs[i]
        col=colors[i]
        size=s_dot*5/3
        axes.scatter(np.real(freq_i),order_i,marker='s',
        color=col,s=size,label='Mode '+str(i+1))
        axes.plot(np.real(freq_i),order_i,
        color=col,linewidth=size/20)
       
    axes.legend()
    axes.set_xlim(xmin,xmax)
    axes.set_ylabel('Modal order')
    axes.set_xlabel('Frequencies [rad/s]')
    plt.tight_layout() 
      
    return figur_h    