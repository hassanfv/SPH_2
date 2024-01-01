
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

  uEvolution = np.zeros((N_T, N_nH, N_Z))
  TEvolution = np.zeros((N_T, N_nH, N_Z))
  muArr = np.zeros((N_T, N_nH, N_Z))
  
  metalz = np.zeros((N_T, N_nH, N_Z, 14))
  
  uArr = np.zeros((N_T, N_nH, N_Z))

  for ndx_nH in range(n_densities):
    for ndx_Z in range(n_metallicities):
      for ndx_T in range(n_temperatures):        
      
        T0 = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, 0]
      
        T1 = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, -2]
        T2 = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, -1]
        dT = T2 - T1
        TEvolution[ndx_T, ndx_nH, ndx_Z] = dT
        
        Ab = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, -1]
      
        utmp0, mu0 = Temp_to_u(T0, Ab, ndx_T, ndx_nH, ndx_Z)
      
        utmp1, mu = Temp_to_u(T1, Ab, ndx_T, ndx_nH, ndx_Z)
        utmp2, mu = Temp_to_u(T2, Ab, ndx_T, ndx_nH, ndx_Z)
        dutmp = utmp2 - utmp1
        uEvolution[ndx_T, ndx_nH, ndx_Z] = dutmp
        muArr[ndx_T, ndx_nH, ndx_Z] = mu
        
        uArr[ndx_T, ndx_nH, ndx_Z] = utmp0 # uArr will be used to find N_T in the main SPH code!
        
        #print(f'T0 = {T0},   u0 = {utmp0:.4E}')
        
        #print()
        #print(ndx_T, ndx_nH, ndx_Z, T1, T2, dT, dutmp)
        
        #        HI  HII  CI  CII  CII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII
        nxIDz = [1,  2,   7,  8,   9,   10,  58,   59,    60,   19, 28,  111,  44,  45]
        IDz = ['HI', 'HII', 'CI', 'CII', 'CII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']
        for ii in range(len(IDz)):
          metalz[ndx_T, ndx_nH, ndx_Z, ii] = Ab[nxIDz[ii]]

  return uEvolution, TEvolution, uArr, muArr, metalz



gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']


dirX = './resBin0.10/' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#------- This is only needed for getting N_nH, N_T, N_Z, etc --------
f = h5py.File(dirX + 'grid_noneq_evolution_0.2kpc.hdf5', 'r')

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

timeArr_in_sec = f['TimeArray_seconds'][:]
timeArr_yrs = timeArr_in_sec/3600/24/365.25
#print(f'timeArr_in_sec = ', timeArr_in_sec)
#print(f'timeArr_in_yrs = ', timeArr_yrs) # in yrs
print()

TT = time.time()

kpcs = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'] #!!!!!!!!!!!!!!! Update this if needed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
kpcsF = [np.float32(tmp) for tmp in kpcs]
N_kpc = len(kpcs)

uEvolutionX = np.zeros((N_kpc, N_T, N_nH, N_Z))
TEvolutionX = np.zeros((N_kpc, N_T, N_nH, N_Z))
muArrX = np.zeros((N_kpc, N_T, N_nH, N_Z))
metalzX = np.zeros((N_kpc, N_T, N_nH, N_Z, 14))
uArrX = np.zeros((N_kpc, N_T, N_nH, N_Z))


filz = np.sort(glob.glob(dirX + '*.hdf5'))

for i in range(N_kpc):

  f = h5py.File(dirX + 'grid_noneq_evolution_' + kpcs[i] + 'kpc.hdf5', 'r')

  AbundanceEvolution = f['AbundanceEvolution'][:]
  densities = f['TableBins/Densities'][:]
  metallicities = f['TableBins/Metallicities'][:]
  temperatures = f['TableBins/Temperatures'][:]
  TemperatureEvolution = f['TemperatureEvolution'][:]

  uEvolution, TEvolution, uArr, muArr, metalz = h_func(N_T, N_nH, N_Z)    

  uEvolutionX[i, :, :, :] = uEvolution
  TEvolutionX[i, :, :, :] = TEvolution
  muArrX[i, :, :, :] = muArr
  metalzX[i, :, :, :, :] = metalz
  
  uArrX[i, :, :, :] = uArr

print(uEvolutionX.shape, muArrX.shape, metalzX.shape)


N_M = len(metalzX[0, 0, 0, 0, :])

print(f'N_kpc = {N_kpc}, N_nH = {N_nH}, N_Z = {N_Z}, N_T = {N_T}, N_M = {N_M}')

print('uEvol = ', uEvolutionX[2, 5, 6, 1])
print('TEvol = ', TEvolutionX[2, 5, 6, 1])
print(f'uArr = {uArrX[2, 5, 6, 1]:.5E}')
print()


densities = np.array([10**tmp for tmp in densities])
metallicities = np.array([10**tmp for tmp in metallicities])

dictx = {'densities': densities, 'metallicities': metallicities, 'temperatures': temperatures, 'timeArr_in_sec': timeArr_in_sec, 'kpc': kpcsF,
         'uEvolution': uEvolutionX, 'TEvolution': TEvolutionX, 'uArr': uArrX, 'muArr': muArrX, 'metalz': metalzX}
with open('coolHeatGridDec2023.pkl', 'wb') as f:
  pickle.dump(dictx, f)

with open('coolHeatGridDec2023.bin', 'wb') as f:
  # Save N_Kpc, N_T, N_nH, N_Z, N_M for dimensions
  f.write(struct.pack('i', N_kpc))
  f.write(struct.pack('i', N_T))
  f.write(struct.pack('i', N_nH))
  f.write(struct.pack('i', N_Z))
  f.write(struct.pack('i', N_M))
  
  # Save kpc
  for val in kpcsF:
    f.write(struct.pack('f', val))  # use 'f' format for float
  
  # Save densities
  for val in densities:
    f.write(struct.pack('f', val))  # use 'f' format for float
      
  # Save metallicities
  for val in metallicities:
    f.write(struct.pack('f', val))  # use 'f' format for float
      
  # Save temperatures
  for val in temperatures:
    f.write(struct.pack('f', val))  # use 'f' format for float
      
  # Save res
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
            f.write(struct.pack('f', uEvolutionX[t][i][j][k]))  # use 'f' format for float

  # Save muA
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
            f.write(struct.pack('f', muArrX[t][i][j][k]))  # use 'f' format for float

  # Save metalz
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
          for ii in range(N_M):
              f.write(struct.pack('f', metalzX[t][i][j][k][ii]))  # use 'f' format for float

  # Save uArr
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
            f.write(struct.pack('f', uArrX[t][i][j][k]))  # use 'f' format for float

print('Elapsed time = ', time.time() - TT)





