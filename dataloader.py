import numpy as np
import matplotlib.pyplot as plt
import nptdms as tdms
import pandas as pd
import time

t0 = time.time()

root_path = "Data/Halogaland_2022_04_22"

tdms_file = tdms.TdmsFile.read("Data/Halogaland_2022_04_22/anode003_2022-02-04-00-00-00Z.tdms")



print(tdms_file.groups())



acceleration = tdms_file['acceleration_data']
print(acceleration.channels())
timestamps = acceleration["timestamp"][:]
print(acceleration["timestamp"][0])
print(len(timestamps))
acc_1y =acceleration["A03-1y"][:]
print(len(acc_1y))

#plt.plot(timestamps, acc_x1)
#plt.show()

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


v = np.zeros_like(acc_1y)
for i in range(len(timestamps)-1):
    v[i] = np.trapz(acc_1y[:i+1], timestamps[:i+1])

plt.plot(timestamps, v)
plt.xlabel("$t$")
plt.ylabel("$v(t)$")
plt.grid()
plt.show()

u = np.zeros_like(acc_1y)
for i in range(len(timestamps)-1):
    u[i] = np.trapz(v[:i+1], timestamps[:i+1])

plt.plot(timestamps, u)
plt.xlabel("$t$")
plt.ylabel("$u(t)$")
plt.grid()
plt.show()



t1 = time.time()

print(str(t1-t0) + " sec")