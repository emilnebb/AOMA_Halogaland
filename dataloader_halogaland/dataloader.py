import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile #docs: https://nptdms.readthedocs.io/en/stable/index.html
import datetime

g = 9.82

class Dataloader:

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