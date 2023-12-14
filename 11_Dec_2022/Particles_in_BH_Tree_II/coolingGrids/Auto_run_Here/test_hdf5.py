
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
def Temp_to_u(T, Ab, ndx_T, ndx_nH, ndx_Z):

  s = 0.0
  p = 0.0
  for j in range(157):
    s += Ab[j] * AtomicMass[j]
    p += Ab[j] # Note that ne is also included in the sum!!

  mu = s / p

  utmp = kB / mH / (gamma - 1.) / mu * T
  
  return utmp, mu



#----- h_func
def h_func(N_T, N_nH, N_Z):

  uEvolution = np.zeros((N_T, N_nH, N_Z, 2)) # 2 is to save the initial and the final u!!!!
  muArr = np.zeros((N_T, N_nH, N_Z, 2))
  
  metalz = np.zeros((N_T, N_nH, N_Z, 14, 2))

  for ndx_nH in range(n_densities):
    for ndx_Z in range(n_metallicities):
      for ndx_T in range(n_temperatures):
      
        T = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, -1]
        Ab = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, -1]
      
        utmp, mu = Temp_to_u(T, Ab, ndx_T, ndx_nH, ndx_Z)
        
        uEvolution[ndx_T, ndx_nH, ndx_Z, 1] = utmp
        muArr[ndx_T, ndx_nH, ndx_Z, 1] = mu
        
        #        HI  HII  CI  CII  CII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII
        nxIDz = [1,  2,   7,  8,   9,   10,  58,   59,    60,   19, 28,  111,  44,  45]
        IDz = ['HI', 'HII', 'CI', 'CII', 'CII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']
        for ii in range(len(IDz)):
          metalz[ndx_T, ndx_nH, ndx_Z, ii, 1] = Ab[nxIDz[ii]]
        
        #------- Initial u values (i.e. u at time = 0) -------
        T_0 = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, 0]
        Ab_0 = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, 0]
        u_0, mu_0 = Temp_to_u(T_0, Ab_0, ndx_T, ndx_nH, ndx_Z)
        
        uEvolution[ndx_T, ndx_nH, ndx_Z, 0] = u_0
        muArr[ndx_T, ndx_nH, ndx_Z, 0] = mu_0
        
        for ii in range(len(IDz)):
          metalz[ndx_T, ndx_nH, ndx_Z, ii, 0] = Ab_0[nxIDz[ii]]

  return uEvolution, muArr, metalz



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

f = h5py.File('./grid_noneq_evolution_0.3kpc_01yrs.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
  print(name)
  for key, val in obj.attrs.items():
    print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
print(TemperatureEvolution.shape)
print('T Evolution original = ', TemperatureEvolution[40, 41, 1, :])

N_nH = n_densities = f['TableBins/N_Densities'][()]
print("N_Densities:", n_densities)

N_Z = n_metallicities = f['TableBins/N_Metallicities'][()]
print("N_Metallicities:", n_metallicities)

N_T = n_temperatures = f['TableBins/N_Temperatures'][()]
print("N_Temperatures:", n_temperatures)
print()

AbundanceEvolution = f['AbundanceEvolution'][:]

uEvolution, muArr, metalz = h_func(N_T, N_nH, N_Z)


#print(uEvolution.shape)
#print('mu = ', muArr[50, 41, 1, :])
#print(uEvolution[40, 41, 1, :])




