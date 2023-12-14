
# USE MPI !
# Note that we should run this code for all grid files for different distances of 0.1kpc, 0.2kpc, 0.3kpc, ... etc.

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from mpi4py import MPI
from numba import njit
import struct

'''
AbundanceEvolution
TableBins
TemperatureEvolution
TimeArray_seconds
'''

def h_func(N_T, N_nH, N_Z):

  uEvolution = np.zeros((N_T, N_nH, N_Z))
  muArr = np.zeros((N_T, N_nH, N_Z))
  
  metalz = np.zeros((N_T, N_nH, N_Z, 14))

  for ndx_nH in range(n_densities):
    for ndx_Z in range(n_metallicities):
      for ndx_T in range(n_temperatures):
      
        T = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, -1]
        
        Ab = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, -1]
        s = 0.0
        p = 0.0
        for j in range(157):
          s += Ab[j] * AtomicMass[j]
          p += Ab[j] # Note that ne is also included in the sum!!

        mu = s / p

        utmp = kB / mH / (gamma - 1.) / mu * T
        uEvolution[ndx_T, ndx_nH, ndx_Z] = utmp
        muArr[ndx_T, ndx_nH, ndx_Z] = mu
        
        #        HI  HII  CI  CII  CII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII
        nxIDz = [1,  2,   7,  8,   9,   10,  58,   59,    60,   19, 28,  111,  44,  45]
        IDz = ['HI', 'HII', 'CI', 'CII', 'CII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']
        for ii in range(len(IDz)):
          metalz[ndx_T, ndx_nH, ndx_Z, ii] = Ab[nxIDz[ii]]

  return uEvolution, muArr, metalz

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#nCPUs = comm.Get_size()

#N = 51 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE ACCORDINGLY !!!!! N is the number of values in the time array!!!!!!!!

#------- used in MPI --------
'''
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
'''
#----------------------------

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']

dist = '0.2' # kpc
tiMe = '08' # yrs

f = h5py.File('grid_noneq_evolution_' + dist + 'kpc_' + tiMe + 'yrs' + '.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
  print(name)
  for key, val in obj.attrs.items():
    print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
#print("TemperatureEvolution: ", TemperatureEvolution)
#print()
print(TemperatureEvolution.shape)
print()


print(TemperatureEvolution[22, 20, 1, :])

s()


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


TT = time.time()

uEvolution, muArr, metalz = h_func(N_T, N_nH, N_Z)

print('uEvolution.shape = ', uEvolution.shape)
print('muArr.shape = ', muArr.shape)
print('metalz.shape = ', metalz.shape)




