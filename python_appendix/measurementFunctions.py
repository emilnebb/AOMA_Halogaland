import numpy as np
def transform(sensor1, sensor2, B):
    """
    Arguments:
    sensor1 : data from sensor 1 in the sensor pair
    sensor2 : data from sensor 2 in the sensor pair
    B       : width of the deck
    Returns:
    acc_trans : the transformed data, array with size Nx3, where first column 
                is y, second column is z and third column is theta     
    ------------------------------
    Function transforms the accelerations from a sensor pair into one component
    in y, z and theta direction
    """
    acc_trans = np.zeros((np.max(np.shape(sensor1)),3))
    acc_trans[:,0] = (sensor1[:,1] + sensor2[:,1])/2
    acc_trans[:,1] = (sensor1[:,2] + sensor2[:,2])/2
    acc_trans[:,2] = (-sensor1[:,2]+sensor2[:,2])/B
    
    return acc_trans

def stdAccMeasurements(data_acc):
    """
    Arguments:
    data_acc : acceleration data , units m/s^2 
    Returns:
    sigma_y, sigma_z, sigma_theta : standard deviation in y, z and theta direction
    ----------------------------------
    This function is made for both input from one sensor pair, and from all
    sensor pairs. It takes the acceleration data and calculates the standard deviation
    """
    B = 18.6 #width of bridge deck
    
    data_acc=data_acc-np.mean(data_acc,axis=0)  # Removing the mean from the series
    acc_trans = np.zeros([data_acc.shape[0], int(data_acc.shape[1]/2)])
    k=0
    for i in range(0,data_acc.shape[1],6):
        acc_trans[:,k:k+3] = transform(data_acc[:,i:i+3], data_acc[:,i+3:i+6],B) #Transforming into one component per sensor pair
        k +=3
    #Separating into each component           
    acc_y = acc_trans[:,np.arange(0,acc_trans.shape[1], 3)] #Acceleration in y-direction
    acc_z = acc_trans[:,np.arange(1,acc_trans.shape[1], 3)] #Acceleration in z-direction
    acc_theta = acc_trans[:,np.arange(2,acc_trans.shape[1], 3)] #Acceleration in theta-direction
    # Computing the standard deviation
    sigma_y = np.std(acc_y, axis=0)
    sigma_z= np.std(acc_z, axis=0)
    sigma_theta = np.std(acc_theta, axis=0)
    
    return sigma_y,sigma_z, sigma_theta 
    