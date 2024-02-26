import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from numba import njit
import struct
import glob

'''
AbundanceEvolution
TableBins
TemperatureEvolution
TimeArray_seconds
'''


f = h5py.File('grid_noneq_evolution_0.1kpc.hdf5', 'r')
#f = h5py.File('./10yrs_veryFine/grid_noneq_evolution_0.3kpc.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
  print(name)
  for key, val in obj.attrs.items():
    print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
print(TemperatureEvolution.shape)

t_Arr_in_sec = f['TimeArray_seconds'][:]
t_Arr_in_yrs = t_Arr_in_sec / (3600. * 24. * 365.25)

N_nH = n_densities = f['TableBins/N_Densities'][()]
print("N_Densities:", n_densities)

N_Z = n_metallicities = f['TableBins/N_Metallicities'][()]
print("N_Metallicities:", n_metallicities)

N_T = n_temperatures = f['TableBins/N_Temperatures'][()]
print("N_Temperatures:", n_temperatures)
print()
densities = f['TableBins/Densities'][:]
metallicities = f['TableBins/Metallicities'][:]
temperatures = f['TableBins/Temperatures'][:]

AbundanceEvolution = f['AbundanceEvolution'][:]

print('temperatures = ', temperatures)
print()
print('nH = ', densities)

timeArr_in_sec = f['TimeArray_seconds'][:]
timeArr_yrs = timeArr_in_sec / 3600/24/365.25
N_t = n_time = len(timeArr_in_sec)
print(f'N_time = {N_t}')
print()

print('AbundanceEvolution.shape = ', AbundanceEvolution.shape) # [Temp, nH, Z, metals, time]
print()

inH = 6
print('nH = ', 10**densities[inH])
iTemp = 2
print('T = ', 10**temperatures[iTemp])
iZ = 1
print('Z = ', metallicities[iZ])
it = -1
print('time = ', timeArr_yrs[it])
print()
print(f'T after {timeArr_yrs[it]:.2f} years is, T = {TemperatureEvolution[iTemp, inH, iZ, it]:.2f} K.')
print()

print('T Evolution = ', TemperatureEvolution[iTemp, inH, iZ, :])
print()

AllCarbon = AbundanceEvolution[iTemp, inH, iZ, 7:15, it]
print(np.log10(10*sum(AllCarbon))) # Multiplied by 10 because we had assumed Z = 0.1 Z_sun!
print()

print('HI = ', AbundanceEvolution[iTemp, inH, iZ, 1, it])
print('HII = ', AbundanceEvolution[iTemp, inH, iZ, 2, it])

print('log(HI) = ', np.log10(AbundanceEvolution[iTemp, inH, iZ, 1, it]))
print('log(HII) = ', np.log10(AbundanceEvolution[iTemp, inH, iZ, 2, it]))

print('HI + HII = ', AbundanceEvolution[iTemp, inH, iZ, 1, it] + AbundanceEvolution[iTemp, inH, iZ, 2, it])
print('CI = ', AbundanceEvolution[iTemp, inH, iZ, 7, it])
print('CII = ', AbundanceEvolution[iTemp, inH, iZ, 8, it])
print('CIII = ', AbundanceEvolution[iTemp, inH, iZ, 9, it])
print('CIV = ', AbundanceEvolution[iTemp, inH, iZ, 10, it])

totC = (np.sum(AbundanceEvolution[iTemp, inH, iZ, 7:15, it]))
print()
print('sum (C) = ', np.sum(totC))
print('log(sum (C)) = ', np.log10(np.sum(totC)))
print()
print('total C ratio = ', np.log10(1e-30+AbundanceEvolution[iTemp, inH, iZ, 7:15, it]/totC))
print()
print('total C ratio = ', (AbundanceEvolution[iTemp, inH, iZ, 7:15, it]/totC))






