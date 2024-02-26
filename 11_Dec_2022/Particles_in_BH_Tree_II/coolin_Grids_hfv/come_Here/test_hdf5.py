
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
        
        #if T < 500:
        #  print('XXXX = ', T, mu, utmp)
        
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

f = h5py.File('grid_noneq_evolution_1.0kpc.hdf5', 'r')

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

print('temperatures = ', temperatures)
#s()

inH = 6
print('nH = ', 10**densities[inH])
iTemp = 10
print('T = ', 10**temperatures[iTemp])
iZ = 1
print('Z = ', metallicities[iZ])

#TEvol = TemperatureEvolution[iTemp, inH, iZ, :]
#TEvol = [np.round(tmp, 2) for tmp in TEvol]

#print(TEvol)

#for i in range(len(TEvol)):  
#  print((TEvol[i+1]-TEvol[i])/TEvol[i])

AbundanceEvolution = f['AbundanceEvolution'][:]

#print('TemperatureEvolution = ', TemperatureEvolution)

uEvolution, muArr, metalz = h_func(N_T, N_nH, N_Z)

#print('T Evolution original = ', TemperatureEvolution[iTemp, inH, iZ, :])

N_time = len(TemperatureEvolution[iTemp, inH, iZ, :])

print()
print()



jmp = 1

for i in range(0, N_time-jmp, jmp):
  print(t_Arr_in_yrs[i+1], TemperatureEvolution[iTemp, inH, iZ, i+1], TemperatureEvolution[iTemp, inH, iZ, i])
  #print(t_Arr_in_yrs[i+1], TemperatureEvolution[iTemp, inH, iZ, i+jmp]-TemperatureEvolution[iTemp, inH, iZ, i])




