import numpy as np
from nptdms import TdmsFile #docs: https://nptdms.readthedocs.io/en/stable/index.html
import h5py
from dataloader_halogaland.processer import low_pass, downsample

g = 9.82

class TDMS_dataloader:
    """
    A dataloader specified for the data logged at Hålogaland bridge, loaded from TDMS file format.
    """

    def __init__(self, path: str):

        self.path = path
        self.anodes = ['/anode003', '/anode004', '/anode005', '/anode006', '/anode007', '/anode008',
                       '/anode009', '/anode010']  # List of all data loggers
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

    def __init__(self, path: str, bridgedeck_only: bool):

        self.path = path
        self.data_types = None
        self.hdf5_file = None
        self.hdf5_file = h5py.File(self.path, 'r')
        self.periods = list(self.hdf5_file.keys())
        self.data_types = list(self.hdf5_file[self.periods[0]].keys())

        if bridgedeck_only:
            self.acceleration_sensors = ['A03-1', 'A03-2', 'A04-1', 'A04-2', 'A05-1', 'A05-2', 'A06-1', 'A06-2',
                                         'A07-1', 'A07-2', 'A08-1', 'A08-2', 'A09-1',
                                         'A09-2', 'A10-1', 'A10-2']  # bridge deck only
        else:
            self.acceleration_sensors = ['A01-1', 'A03-1', 'A03-2', 'A04-1', 'A04-2', 'A05-1', 'A05-2', 'A06-1',
                                         'A06-2', 'A06-3', 'A06-4', 'A07-1', 'A07-2', 'A08-1', 'A08-2', 'A08-3',
                                         'A08-4', 'A09-1', 'A09-2', 'A10-1', 'A10-2', 'A11-1'] # all acc-sensors

        self.wind_sensors = ['W03-7-1', 'W04-15-1', 'W05-17-1', 'W05-18-1', 'W05-19-1', 'W05-19-2', 'W07-28-1',
                             'W10-45-1', 'W10-47-1', 'W10-49-1']

        self.temp_sensors = ['T01-1', 'T01-2', 'T02-1', 'T02-2', 'T03-1', 'T03-2', 'T04-1', 'T04-2', 'T05-1',
                             'T05-2', 'T06-1', 'T06-2', 'T07-1', 'T07-2', 'T08-1', 'T08-2', 'T09-1', 'T09-2',
                             'T10-1', 'T10-2', 'T11-1', 'T11-2']

        #print("Available accelerometers: " + str(self.acceleration_sensors))

    def load_acceleration(self, period: str, sensor: str, axis: str, preprosess=False, cutoff_frequency = None,
                          filter_order=None):
        # TODO: write function description

        acc_data = self.hdf5_file[period][self.data_types[0]][sensor][axis]

        if preprosess:
            sampling_rate = self.hdf5_file[period][self.data_types[0]][sensor].attrs['samplerate']
            filtered_acc = low_pass(acc_data - np.mean(acc_data), sampling_rate, cutoff_frequency, filter_order)
            acc_data = downsample(sampling_rate, filtered_acc, cutoff_frequency*2)

        return acc_data

    def load_all_acceleration_data(self, period: str, preprosess=False, cutoff_frequency = None, filter_order=None):
        #TODO: write function description

        #Check if all channels are included
        if not set(self.acceleration_sensors).issubset(list(self.hdf5_file[period][self.data_types[0]].keys())):
            return False

        acc_example = self.load_acceleration(self.periods[12], self.acceleration_sensors[0], 'x', preprosess,
                                             cutoff_frequency, filter_order)

        acc_x = np.zeros((len(acc_example), len(self.acceleration_sensors)))
        acc_y = np.zeros((len(acc_example), len(self.acceleration_sensors)))
        acc_z = np.zeros((len(acc_example), len(self.acceleration_sensors)))

        counter = 0
        for sensor in self.acceleration_sensors:
            acc_x[:, counter] = self.load_acceleration(period, sensor, 'x', preprosess, cutoff_frequency, filter_order)
            acc_y[:, counter] = self.load_acceleration(period, sensor, 'y', preprosess, cutoff_frequency, filter_order)
            acc_z[:, counter] = self.load_acceleration(period, sensor, 'z', preprosess, cutoff_frequency, filter_order)
            counter += 1

        acc_matrix = np.concatenate((acc_x, acc_y, acc_z), axis=1)

        return acc_matrix

    def load_wind(self, period: str, sensor: str):
        #TODO: write function descritpion

        #Wind measurements has a 32 Hz sampling rate

        wind_magnitude = np.array(self.hdf5_file[period]['wind'][sensor]['magnitude'])
        wind_direction = np.array(self.hdf5_file[period]['wind'][sensor]['direction'])

        return wind_magnitude, wind_direction

    def load_wind_stat_data(self, period: str, timeseries_length: int, timeseries_num: int):
        #TODO: write function description

        # Check if all channels are included
        if not set(self.wind_sensors).issubset(list(self.hdf5_file[period]['wind'].keys())):
            return False

        #Make an assumption that wind sensor at mid span of the bridge makes up a fairly good representation
        #of the overall wind magnitude along the bridge span

        wind_magnitude, wind_direction = self.load_wind(period, 'W07-28-1')

        fs = self.hdf5_file[period]['wind']['W07-28-1'].attrs['samplerate']

        time_series_wind_magnitude = wind_magnitude[timeseries_num*timeseries_length*fs*60:(timeseries_num+1)
                                                                                      *timeseries_length*fs*60]
        time_series_wind_direction = wind_direction[timeseries_num*timeseries_length*fs*60:(timeseries_num+1)
                                                                                      *timeseries_length*fs*60]
        mean_wind_speed = np.mean(time_series_wind_magnitude)
        max_wind_speed = np.max(time_series_wind_magnitude)
        mean_wind_direction = np.mean(time_series_wind_direction)

        return mean_wind_speed, max_wind_speed, mean_wind_direction

    def load_temp(self, period: str, sensor: str):
        #TODO: write function description

        #Temperature measurements has a 0.25 Hz sampling rate

        temp_data = np.array(self.hdf5_file[period]['temperature'][sensor]['x'])

        return temp_data

    def load_temp_stat_data(self, period: str, timeseries_length: int, timeseries_num: int):
        # TODO: write function description

        # Check if all channels are included
        if not set(self.temp_sensors).issubset(list(self.hdf5_file[period]['temperature'].keys())):
            return False

        #Pick one temp sensor to collect data from
        all_temp_data = self.load_temp(period, 'T07-1')

        fs = self.hdf5_file[period]['temperature']['T07-1'].attrs['samplerate']

        time_series_temp_data = all_temp_data[timeseries_num * timeseries_length * int(fs * 60):(timeseries_num + 1)
                                                                                * timeseries_length * int(fs * 60)]
        mean_temp = np.mean(time_series_temp_data)

        return mean_temp

class Mode:
    # TODO: write class description
    def __init__(self, frequency, mode_shape, damping=None, mode_label=None):
        self.frequency = frequency
        self.damping = damping
        self.mode_shape = mode_shape
        self.mode_label = mode_label


class HDF5_result_loader:
    """
    A dataloader specified to load results from AOMA analysis stored in a h5 format.
    """

    def __init__(self, path: str):
        self.path = path
        self.hdf5_file = h5py.File(self.path, 'r')
        self.periods = list(self.hdf5_file.keys())
        self.features = ['Damping', 'Frequencies', 'Modeshape']

    def get_modes_in_period(self, period):
        # TODO: write function description

        freqs = np.array(self.hdf5_file[period]['Frequencies'])
        dampings = np.array(self.hdf5_file[period]['Damping'])
        mode_shapes = np.array(self.hdf5_file[period]['Modeshape'])

        modes_in_period = []

        for i in range(len(freqs)):
            mode = Mode(freqs[i], mode_shapes[i], dampings[i])
            modes_in_period.append(mode)

        return np.array(modes_in_period)

    def get_modes_all_periods(self):
        # TODO: write function description

        all_modes = []

        for period in self.periods:
            all_modes.append(self.get_modes_in_period(period))

        return all_modes

    def get_statistics(self):
        # TODO: write function description
        temps = []
        mean_wind_speed = []
        max_wind_speed = []
        mean_wind_direction = []
        execution_time = []

        for period in self.periods:
            temps.append(self.hdf5_file[period].attrs['Mean temp'])
            mean_wind_speed.append(self.hdf5_file[period].attrs['Mean wind speed'])
            max_wind_speed.append(self.hdf5_file[period].attrs['Max wind speed'])
            mean_wind_direction.append(self.hdf5_file[period].attrs['Mean wind direction'])
            execution_time.append(self.hdf5_file[period].attrs['Execution time'])

        temps = np.array(temps)
        mean_wind_speed = np.array(mean_wind_speed)
        max_wind_speed = np.array(max_wind_speed)
        mean_wind_direction = np.array(mean_wind_direction)
        execution_time = np.array(execution_time)

        return temps, mean_wind_speed, max_wind_speed, mean_wind_direction, execution_time

class FEM_result_loader:

    def __init__(self, path: str):
        self.path = path
        self.hf = h5py.File(self.path, 'r')

        # manually picked bridge deck modes from Abaqus model
        self.deck_modes_idx = np.array([1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 25, 32, 33,
                                   34, 36, 37, 40, 45, 46, 48, 49, 50]) - 1

        self.sensor_labels = ['3080_U1', '2080_U1', '3140_U1', '2140_U1', '3200_U1',
                                 '2200_U1', '3240_U1', '2240_U1', '3290_U1', '2290_U1',
                                 '3340_U1', '2340_U1', '3420_U1', '2420_U1', '3500_U1',
                                 '2500_U1',
                                 '3080_U2', '2080_U2', '3140_U2', '2140_U2', '3200_U2',
                                 '2200_U2', '3240_U2', '2240_U2', '3290_U2', '2290_U2',
                                 '3340_U2', '2340_U2', '3420_U2', '2420_U2', '3500_U2',
                                 '2500_U2',
                                 '3080_U3', '2080_U3', '3140_U3', '2140_U3', '3200_U3',
                                 '2200_U3', '3240_U3', '2240_U3', '3290_U3', '2290_U3',
                                 '3340_U3', '2340_U3', '3420_U3', '2420_U3', '3500_U3',
                                 '2500_U3']

        phi_label_temp = np.array(self.hf.get('phi_label'))
        phi_label = phi_label_temp[:].astype('U10').ravel().tolist()

        sensor_indexes = []
        for label in self.sensor_labels:
            if label in phi_label:
                sensor_indexes.append(phi_label.index(label))

        self.f = np.array(self.hf.get('f'))[self.deck_modes_idx]
        phi = np.array(self.hf.get('phi'))[:, self.deck_modes_idx]
        self.phi = phi[sensor_indexes, :]

    def get_all_modes(self):

        modes = []

        for i in range(len(self.f)):
            mode = Mode(self.f[i], self.phi[:, i])
            modes.append(mode)

        return np.array(modes)

