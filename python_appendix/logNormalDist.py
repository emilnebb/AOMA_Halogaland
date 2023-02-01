import numpy as np
def logNormParam(data):
    '''
    Arguments:
    data : the data that the lognormal parameters are fitted to
    Returns:
    my, sigma : parameters of the lognormal distribution
    ------------------------------------------------------
    Function for finding the distribution parameters for data with a lognormal distribution
    ''' 
    norm = np.log(data)
    my = np.mean(norm)
    sig = np.std(norm)
    
    return my, sig

