
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from numba import njit
import struct

'''
AbundanceEvolution
TableBins
TemperatureEvolution
TimeArray_seconds
'''

#N = 11 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE ACCORDINGLY !!!!! N is the number of values in the time array!!!!!!!!

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']

#dist = '0.2' # kpc

f = h5py.File('0.2kpc_t_5yrs.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
#print("TemperatureEvolution: ", TemperatureEvolution)
#print()
print(TemperatureEvolution.shape)
#print()

# Get the array of Density
N_nH = n_densities = f['TableBins/N_Densities'][()]
print("N_Densities:", n_densities)

N_Z = n_metallicities = f['TableBins/N_Metallicities'][()]
print("N_Metallicities:", n_metallicities)

N_T = n_temperatures = f['TableBins/N_Temperatures'][()]
print("N_Temperatures:", n_temperatures)
print()

AbundanceEvolution = f['AbundanceEvolution'][:]
#print("AbundanceEvolution: ", AbundanceEvolution)
#print()
print(AbundanceEvolution.shape)
#print()

densities = f['TableBins/Densities'][:]
print("Densities array: ", densities)
print()

metallicities = f['TableBins/Metallicities'][:]
print("Metallicities array: ", metallicities)
print()

temperatures = f['TableBins/Temperatures'][:]
print("Temperatures array: ", temperatures)
print()

timeArr = f['TimeArray_seconds'][:]
timeArr_Myrs = timeArr/3600/24/365.25/1e6
timeArr_kyrs = timeArr/3600/24/365.25/1e3
timeArr_yrs = timeArr/3600/24/365.25
print(f'timeArr_in_sec = ', timeArr)
#print(f'timeArr_in_kyrs = ', timeArr/3600/24/365.25/1e3) # in kyrs
#print(f'timeArr_in_Myrs = ', timeArr/3600/24/365.25/1e6) # in Myrs
print()

#print('TemperatureEvolution = ', TemperatureEvolution)

N_time = len(timeArr)

print('timeArr.shape = ', timeArr.shape)


uEvolution = np.zeros_like(TemperatureEvolution)

print(uEvolution.shape)

print()
print()

n_T = np.where(temperatures == 7)[0][0]
n_nH = np.where(densities == 1)[0][0]
n_Z = 1

print('temperatures[n_T], densities[n_nH], metallicities[n_Z] = ', temperatures[n_T], densities[n_nH], metallicities[n_Z])

print(TemperatureEvolution[n_T, n_nH, n_Z, :])







