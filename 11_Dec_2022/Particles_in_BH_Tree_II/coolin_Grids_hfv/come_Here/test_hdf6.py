
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

#----- Temp_to_u
def Temp_to_u(T, Ab):

  s = 0.0
  p = 0.0
  for j in range(157):
    s += Ab[j] * AtomicMass[j]
    p += Ab[j] # Note that ne is also included in the sum!!

  mu = s / p

  utmp = kB / mH / (gamma - 1.) / mu * T
  
  return utmp, mu



#----- h_func
def h_func(N_T, N_nH, N_Z, N_t):

  uEvolution = np.zeros((N_T, N_nH, N_Z, N_t))
  muArr = np.zeros((N_T, N_nH, N_Z, N_t))
  
  metalz = np.zeros((N_T, N_nH, N_Z, 14, N_t))
  
  uArr = np.zeros((N_T, N_nH, N_Z))

  for ndx_nH in range(n_densities):
    for ndx_Z in range(n_metallicities):
      for ndx_T in range(n_temperatures):
        for ndx_t in range(n_time):      
        
          Ttmp = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, ndx_t]
          Ab = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, ndx_t]
        
          utmp, mu = Temp_to_u(Ttmp, Ab)

          uEvolution[ndx_T, ndx_nH, ndx_Z, ndx_t] = utmp
          muArr[ndx_T, ndx_nH, ndx_Z, ndx_t] = mu
          
          if ndx_t == 0:
            uArr[ndx_T, ndx_nH, ndx_Z] = utmp # uArr will be used to find N_T in the main SPH code!
          
          #        HI  HII  CI  CII  CII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII
          nxIDz = [1,  2,   7,  8,   9,   10,  58,   59,    60,   19, 28,  111,  44,  45]
          IDz = ['HI', 'HII', 'CI', 'CII', 'CII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']
          for ii in range(len(IDz)):
            metalz[ndx_T, ndx_nH, ndx_Z, ii, ndx_t] = Ab[nxIDz[ii]]

  return uEvolution, uArr, muArr, metalz



gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']

#!!!!!!!!!!!!! You may need to update these values !!!!!! This is only needed for getting N_nH, N_T, N_Z, etc !!!!!!!
dist = '0.3' # kpc
tiMe = '01' # yrs
#f = h5py.File('./' + dist + 'kpc/' + 'grid_noneq_evolution_' + dist + 'kpc_' + tiMe + 'yrs' + '.hdf5', 'r')

f = h5py.File('grid_noneq_evolution_0.1kpc.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
  print(name)
  for key, val in obj.attrs.items():
    print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
print(TemperatureEvolution.shape)
#print('T Evolution original = ', TemperatureEvolution[40, 41, 1, :])

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

inH = 53
print('nH = ', 10**densities[inH])
iTemp = 45
print('T = ', 10**temperatures[iTemp])
iZ = 1
print('Z = ', metallicities[iZ])
it = 3
print('time = ', timeArr_yrs[it])
print()

uEvolution, uArr, muArr, metalz = h_func(N_T, N_nH, N_Z, N_t)

print(f'u before evolution = {uEvolution[iTemp, inH, iZ, 0]:.5E}')
print(f'u after {timeArr_yrs[it]} yrs of evolution = {uEvolution[iTemp, inH, iZ, it]:.5E}')
print(f'mu = {muArr[iTemp, inH, iZ, it]}')
print(f'HI fraction = ', metalz[iTemp, inH, iZ, 0, it])

print()




