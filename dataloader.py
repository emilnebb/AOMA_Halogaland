import numpy as np
import matplotlib.pyplot as plt
import nptdms as tdms
import pandas as pd
import time

t0 = time.time()

root_path = "Data/Halogaland_2022_04_22"

"""
with tdms.TdmsFile.open("Data/Halogaland_2022_04_22/anode003_2022-02-04-00-00-00Z.tdms") as tdms_file:
    acceleration = tdms_file['acceleration_data']
    acc_dict = {}
    acc_dict['timestamp'] = acceleration['timestamp'][:]
    acc_dict['1x'] = acceleration['A03-1x'][:]
    acc_dict['1y'] = acceleration['A03-1y'][:]
    acc_dict['1z'] = acceleration['A03-1z'][:]
    acc_dict['2x'] = acceleration['A03-2x'][:]
    acc_dict['2y'] = acceleration['A03-2y'][:]
    acc_dict['2z'] = acceleration['A03-2z'][:]
    acc_dict['acc'] = acceleration['acc06'][:]

t1 = time.time()

print(str(t1-t0) + " sec")
"""


t0 = time.time()
tdms_file = tdms.TdmsFile.read("Data/Halogaland_2022_04_22/anode003_2022-02-04-00-00-00Z.tdms")

t1 = time.time()

print(str(t1-t0) + " sec")
print(tdms_file.groups())
print(tdms_file['acceleration_data'].channels())
print(tdms_file['strain_data'].channels())

#timestamps = acceleration["timestamp"][:]
#print(acc_dict)
#print(len(timestamps))
#print(type(timestamps))
#acc_1y =acc_dict["1x"][:]
#print(len(acc_1y))
#print(acc_1y[0])
#plt.plot(timestamps, acc_x1)
#plt.show()


"""
a_g_fft = np.fft.fft(acc_1y)
freq = np.fft.fftfreq(timestamps.shape[-1])
print(max(a_g_fft.real))
print(max(freq))
plt.plot(freq, a_g_fft.real)
plt.xlim(-0.5,0.5)
plt.xlabel("f [Hz]")
#plt.ylim(0,1000)
plt.ylabel("acceleration amplitude")
plt.grid()
plt.show()
"""




