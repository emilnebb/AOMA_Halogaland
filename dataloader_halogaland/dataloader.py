import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile #docs: https://nptdms.readthedocs.io/en/stable/index.html
import datetime
import h5py
from dataloader_halogaland.processer import low_pass, downsample

g = 9.82

class TDMS_dataloader:
    """
    A dataloader specified for the data logged at Hålogaland bridge, loaded from TDMS file format.
    """

    def __init__(self, path: str):

        self.path = path
        self.anodes = ['/anode003', '/anode004', '/anode005', '/anode006', '/anode007', '/anode008', '/anode009', '/anode010']  # List of all data loggers
        self.acc_names = ['A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
        self.strain_names = ['SG03', 'SG04', 'SG05', 'SG06', 'SG07', 'SG08', 'SG09', 'SG10']
        self.fileToRead = '2022-02-04-00-00-00Z.tdms'

    def read_file(self, anode: str) -> TdmsFile:

        tdms_file = TdmsFile.read(self.path + anode + '_' + self.fileToRead)

        return tdms_file

    def load_acceleration(self, accName: str, tdmsFile: TdmsFile) -> dict:
        """
        Function to read acceleration data from one single TdmsFile
        :param anodeName: The anode to read acceleration from
        :param tdmsFile:  TdmsFile to read acceleration from
        :return: Dictionary with accelerometer data from all sensorpairs with corresponding timestamps,
        acceleration in unit m/s^2
        """

        acceleration = tdmsFile['acceleration_data']
        sensors = ['1x', '1y', '1z', '2x', '2y', '2z']
        acc_dict = {}

        #acc_dict['timestamp'] = np.array([datetime.datetime.utcfromtimestamp(x/1000000000) for x in acceleration['timestamp'][:]])
        acc_dict['timestamp'] = acceleration['timestamp'][:]/1000000000 #convert to seconds


        for i in range(len(sensors)):
            conversion_factor = float(acceleration[accName + '-' + sensors[i]].properties['conversion_factor'])
            acc_dict[sensors[i]] = acceleration[accName + '-' + sensors[i]][:] * g / conversion_factor

        return acc_dict

    def load_strain(self, strainName: str, tdmsFile: TdmsFile) -> dict:
        """
        Function to read strain data from one single TdmsFile
        :param anodeName: The anode to read strain from
        :param tdmsFile:  TdmsFile to read strain from
        :return: Dictionary with strain data from all sensors with corresponding timestamps,
        strain in unit MPa
        """

        strain = tdmsFile['strain_data']
        sensors = ['1', '2', '3', '3']
        strain_dict = {}

        strain_dict['timestamp'] = strain['timestamp'][:]/1000000000 #convert to seconds

        for i in range(len(sensors)):
            conversion_factor = float(strain[strainName + '-' + sensors[i]].properties['conversion_factor'])
            strain_dict[sensors[i]] = strain[strainName + '-' + sensors[i]][:] / conversion_factor

        return strain_dict

class HDF5_dataloader:
    """
    A dataloader specified for the data logged at Hålogaland bridge, loaded from HDF5 file format.
    """

    def __init__(self, path: str):

        self.path = path
        self.periods = None
        self.data_types = None
        self.acceleration_sensors = ['A03-1', 'A04-1', 'A05-1', 'A06-1', 'A07-1', 'A08-1', 'A09-1', 'A10-1', 'A03-2',
                                     'A04-2', 'A05-2', 'A06-2', 'A07-2', 'A08-2', 'A09-2', 'A10-2']
        self.strain_sensors = None
        self.hdf5_file = None
        self.hdf5_file = h5py.File(self.path, 'r')
        self.periods = list(self.hdf5_file.keys())
        self.data_types = list(self.hdf5_file[self.periods[0]].keys())

        #self.acceleration_sensors = list(self.hdf5_file[self.periods[0]][self.data_types[0]].keys())
        self.strain_sensors = list(self.hdf5_file[self.periods[0]][self.data_types[0]].keys())

    def load_acceleration(self, period: str, sensor: str, axis: str, preprosess=False, cutoff_frequency = None, filter_order=None):
        # TODO: write function description

        acc_data = self.hdf5_file[period][self.data_types[0]][sensor][axis]

        if preprosess:
            sampling_rate = self.hdf5_file[period][self.data_types[0]][sensor].attrs['samplerate']
            filtered_acc = low_pass(acc_data - np.mean(acc_data), sampling_rate, cutoff_frequency, filter_order)
            acc_data = downsample(sampling_rate, filtered_acc, cutoff_frequency*2)

        return acc_data

    def load_all_acceleration_data(self, period: str, preprosess=False, cutoff_frequency = None, filter_order=None):
        #TODO: write function description

        acc_example = self.load_acceleration(self.periods[0], self.acceleration_sensors[0], 'x', preprosess, cutoff_frequency, filter_order)

        #acc_matrix = np.zeros((len(acc_example), 48))
        #axis = ['x', 'y', 'z']
        acc_x = np.zeros((len(acc_example), 16))
        acc_y = np.zeros((len(acc_example), 16))
        acc_z = np.zeros((len(acc_example), 16))

        counter = 0
        for sensor in self.acceleration_sensors:
            acc_x[:, counter] = self.load_acceleration(period, sensor, 'x', preprosess, cutoff_frequency, filter_order)
            acc_y[:, counter] = self.load_acceleration(period, sensor, 'y', preprosess, cutoff_frequency, filter_order)
            acc_z[:, counter] = self.load_acceleration(period, sensor, 'z', preprosess, cutoff_frequency, filter_order)
            counter += 1

        acc_matrix = np.concatenate((acc_x, acc_y, acc_z), axis=1)

        return acc_matrix