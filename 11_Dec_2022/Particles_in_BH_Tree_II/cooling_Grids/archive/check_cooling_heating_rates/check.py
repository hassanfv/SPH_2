import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from mpi4py import MPI
from numba import njit
import struct

f = h5py.File('cooling_rates.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

log_cooling_rate = np.array(f["log_cooling_rate"])
log_heating_rate = np.array(f["log_heating_rate"])

print('cooling_rate.shape = ', log_cooling_rate.shape)
print('heating_rate.shape = ', log_heating_rate.shape)
print()


#print('cooling = ', log_cooling_rate)

print(f["TableBins"])

# Access the TableBins group
table_bins_group = f['TableBins']

# Print the names of the members
for name, item in table_bins_group.items():
    print(name)

print()

for name, item in table_bins_group.items():
    print(name)
    if isinstance(item, h5py.Dataset):  # check if the member is a dataset
        print(item[...])  # print its content

Temp = f['TableBins/Temperatures'][()]

#print('Temp = ', Temp)

# cooling_rate.shape =  (71, 9, 5) ===> (T, n, Z)

cool = log_cooling_rate[:, 5, 0]
heat = log_heating_rate[:, 5, 0]

plt.scatter(Temp, cool, s = 5, color = 'k')
plt.scatter(Temp, heat, s = 5, color = 'b')

#plt.ylim(-25, -21.7)

plt.savefig('fig.png')

plt.show()





