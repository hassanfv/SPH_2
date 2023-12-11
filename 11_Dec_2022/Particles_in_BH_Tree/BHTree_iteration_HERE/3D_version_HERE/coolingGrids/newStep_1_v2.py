
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
import glob

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

#!!!!!!!!!!!!! You may need to update these values !!!!!! This is only needed for getting N_nH, N_T, N_Z, etc !!!!!!!
dist = '0.2' # kpc
tiMe = '02' # yrs
f = h5py.File('./' + dist + 'kpc/' + 'grid_noneq_evolution_' + dist + 'kpc_' + tiMe + 'yrs' + '.hdf5', 'r')

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


TT = time.time()

kpcs = ['0.2', '0.4', '0.6'] #!!!!!!!!!!!!!!!!!!!!!!!!! Update this if you have data for more distances !!!!!!!!!!!!!!!
N_kpc = len(kpcs)
k = 0
filz = np.sort(glob.glob('./' + kpcs[k] + 'kpc/*.hdf5'))
yrs = [tmp[-10:-8] for tmp in filz] # Automatically extracting the years as string!!
print(yrs)
N_time = len(yrs)
print('N_time = ', N_time)


timeArr_in_sec = np.array([np.int32(yrs[i]) * 365.25 * 24 * 3600.0 for i in range(len(yrs))])
print('timeArr_in_sec = ', timeArr_in_sec)

kpcsF = np.array([np.float32(kpcs[jj]) for jj in range(len(kpcs))])
print('kpcsF = ', kpcsF) # string converted to float. See kpcs list above.


uEvolutionX = np.zeros((N_kpc, N_T, N_nH, N_Z, N_time))
muArrX = np.zeros((N_kpc, N_T, N_nH, N_Z, N_time))
metalzX = np.zeros((N_kpc, N_T, N_nH, N_Z, 14, N_time))

for k in range(len(kpcs)):

  filz = np.sort(glob.glob('./' + kpcs[k] + 'kpc/*.hdf5'))

  for j in range(len(filz)):

    f = h5py.File(filz[j], 'r')

    AbundanceEvolution = f['AbundanceEvolution'][:]
    densities = f['TableBins/Densities'][:]
    metallicities = f['TableBins/Metallicities'][:]
    temperatures = f['TableBins/Temperatures'][:]

    uEvolution, muArr, metalz = h_func(N_T, N_nH, N_Z)

    uEvolutionX[k, :, :, :, j] = uEvolution
    muArrX[k, :, :, :, j] = muArr
    metalzX[k, :, :, :, :, j] = metalz

N_M = len(metalzX[0, 0, 0, 0, :, 0])

print(f'N_kpc = {N_kpc}, N_nH = {N_nH}, N_Z = {N_Z}, N_T = {N_T}, N_M = {N_M}, N_time = {N_time}')


densities = np.array([10**tmp for tmp in densities])
metallicities = np.array([10**tmp for tmp in metallicities])


dictx = {'densities': densities, 'metallicities': metallicities, 'temperatures': temperatures, 'timeArr_in_sec': timeArr_in_sec, 'kpc': kpcsF,
         'uEvolution': uEvolutionX, 'muArr': muArrX, 'metalz': metalzX}
with open('coolHeatGridNew.pkl', 'wb') as f:
  pickle.dump(dictx, f)


with open('coolHeatGridNew.bin', 'wb') as f:
  # Save N_Kpc, N_nH, N_Z, N_T, N_time for dimensions
  f.write(struct.pack('i', N_kpc))
  f.write(struct.pack('i', N_nH))
  f.write(struct.pack('i', N_Z))
  f.write(struct.pack('i', N_T))
  f.write(struct.pack('i', N_M))
  f.write(struct.pack('i', N_time)) # Note that len(muArr) = N_time!
  
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
      
  # Save timeArr
  for val in timeArr_in_sec:
    f.write(struct.pack('f', val))  # use 'f' format for float
      
  # Save res
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
          for l in range(N_time):
            f.write(struct.pack('f', uEvolutionX[t][i][j][k][l]))  # use 'f' format for float

  # Save muA
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
          for l in range(N_time):
            f.write(struct.pack('f', muArrX[t][i][j][k][l]))  # use 'f' format for float

  # Save metalz
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
          for ii in range(N_M):
            for l in range(N_time):
              f.write(struct.pack('f', metalzX[t][i][j][k][ii][l]))  # use 'f' format for float



print('Elapsed time = ', time.time() - TT)





