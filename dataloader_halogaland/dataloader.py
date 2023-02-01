import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile #docs: https://nptdms.readthedocs.io/en/stable/index.html
from nptdms import TdmsFile

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
        :return: Dictionary with accelerometer data from all sensorpairs with corresponding timestamps
        """

        acceleration = tdmsFile['acceleration_data']
        acc_dict = {}
        acc_dict['timestamp'] = acceleration['timestamp'][:]
        acc_dict['1x'] = acceleration[accName + '-1x'][:]
        acc_dict['1y'] = acceleration[accName + '-1y'][:]
        acc_dict['1z'] = acceleration[accName + '-1z'][:]
        acc_dict['2x'] = acceleration[accName + '-2x'][:]
        acc_dict['2y'] = acceleration[accName + '-2y'][:]
        acc_dict['2z'] = acceleration[accName + '-2z'][:]

        return acc_dict

    def load_strain(self, strainName: str, tdmsFile: TdmsFile) -> dict:
        """
        Function to read strain data from one single TdmsFile
        :param anodeName: The anode to read strain from
        :param tdmsFile:  TdmsFile to read strain from
        :return: Dictionary with strain data from all sensors with corresponding timestamps
        """

        strain = tdmsFile['strain_data']
        strain_dict = {}
        strain_dict['timestamp'] = strain['timestamp'][:]
        strain_dict['SG1'] = strain[strainName + '-1'][:]
        strain_dict['SG2'] = strain[strainName + '-2'][:]
        strain_dict['SG3'] = strain[strainName + '-3'][:]
        strain_dict['SG4'] = strain[strainName + '-4'][:]

        return strain_dict