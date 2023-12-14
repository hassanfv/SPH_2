
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
  TEvolution = np.zeros((N_T, N_nH, N_Z, 2)) # Will be used for debugging!
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
        TEvolution[ndx_T, ndx_nH, ndx_Z, 1] = T
        
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
        TEvolution[ndx_T, ndx_nH, ndx_Z, 0] = T_0
        
        for ii in range(len(IDz)):
          metalz[ndx_T, ndx_nH, ndx_Z, ii, 0] = Ab_0[nxIDz[ii]]

  return uEvolution, muArr, metalz, TEvolution



gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']

#!!!!!!!!!!!!! You may need to update these values !!!!!! This is only needed for getting N_nH, N_T, N_Z, etc !!!!!!!
dist = '0.3' # kpc
tiMe = '03' # yrs
f = h5py.File('./' + dist + 'kpc/' + 'grid_noneq_evolution_' + dist + 'kpc_' + tiMe + 'yrs' + '.hdf5', 'r')

# Print the attributes of HDF5 objects
for name, obj in f.items():
  print(name)
  for key, val in obj.attrs.items():
    print("    %s: %s" % (key, val))
    
TemperatureEvolution = f['TemperatureEvolution'][:]
print(TemperatureEvolution.shape)

N_nH = n_densities = f['TableBins/N_Densities'][()]
print("N_Densities:", n_densities)

N_Z = n_metallicities = f['TableBins/N_Metallicities'][()]
print("N_Metallicities:", n_metallicities)

N_T = n_temperatures = f['TableBins/N_Temperatures'][()]
print("N_Temperatures:", n_temperatures)
print()


TT = time.time()

kpcs = ['0.1', '0.3', '0.5', '0.7', '0.9'] #!!!!!!!!!!!!!!!!!!!!!!!!! Update this if you have data for more distances !!!!!!!!!!!!!!!
N_kpc = len(kpcs)
k = 0
filz = np.sort(glob.glob('./' + kpcs[k] + 'kpc/*.hdf5'))
yrs = [tmp[-10:-8] for tmp in filz] # Automatically extracting the years as string!!
print(yrs)
N_time = len(yrs)
print('N_time = ', N_time)


timeArr_in_sec = np.array([0.0] + [np.int32(yrs[i]) * 365.25 * 24 * 3600.0 for i in range(len(yrs))]) # Note: time 0.0 added manually here!
print('timeArr_in_sec = ', timeArr_in_sec)

kpcsF = np.array([np.float32(kpcs[jj]) for jj in range(len(kpcs))])
print('kpcsF = ', kpcsF) # string converted to float. See kpcs list above.


uEvolutionX = np.zeros((N_kpc, N_T, N_nH, N_Z, N_time+1)) # 1 is added to N_time to have space for time 0!!!!
muArrX = np.zeros((N_kpc, N_T, N_nH, N_Z, N_time+1))
metalzX = np.zeros((N_kpc, N_T, N_nH, N_Z, 14, N_time+1))
TEvolutionX = np.zeros((N_kpc, N_T, N_nH, N_Z, N_time+1)) # used for debugging!

for k in range(len(kpcs)):

  filz = np.sort(glob.glob('./' + kpcs[k] + 'kpc/*.hdf5'))

  for j in range(len(filz)): # each file is for one time!!! so j represents time index!!!

    f = h5py.File(filz[j], 'r')

    AbundanceEvolution = f['AbundanceEvolution'][:]
    densities = f['TableBins/Densities'][:]
    metallicities = f['TableBins/Metallicities'][:]
    temperatures = f['TableBins/Temperatures'][:]
    TemperatureEvolution = f['TemperatureEvolution'][:]

    uEvolution, muArr, metalz, TEvolution = h_func(N_T, N_nH, N_Z)

    if j == 0: # the initial state needs to be imported once!
      uEvolutionX[k, :, :, :, j] = uEvolution[:, :, :, 0]
      muArrX[k, :, :, :, j] = muArr[:, :, :, 0]
      metalzX[k, :, :, :, :, j] = metalz[:, :, :, :, 0]
      TEvolutionX[k, :, :, :, j] = TEvolution[:, :, :, 0]
      
      uEvolutionX[k, :, :, :, j+1] = uEvolution[:, :, :, 1]
      muArrX[k, :, :, :, j+1] = muArr[:, :, :, 1]
      metalzX[k, :, :, :, :, j+1] = metalz[:, :, :, :, 1]
      TEvolutionX[k, :, :, :, j+1] = TEvolution[:, :, :, 1]
      
    else:
      uEvolutionX[k, :, :, :, j+1] = uEvolution[:, :, :, 1]
      muArrX[k, :, :, :, j+1] = muArr[:, :, :, 1]
      metalzX[k, :, :, :, :, j+1] = metalz[:, :, :, :, 1]
      TEvolutionX[k, :, :, :, j+1] = TEvolution[:, :, :, 1]


N_M = len(metalzX[0, 0, 0, 0, :, 0])

N_time += 1 # adding 1 because of the initial state at time zeros (see above)!

print(f'N_kpc = {N_kpc}, N_nH = {N_nH}, N_Z = {N_Z}, N_T = {N_T}, N_M = {N_M}, N_time = {N_time}')


densities = np.array([10**tmp for tmp in densities])
metallicities = np.array([10**tmp for tmp in metallicities])


dictx = {'densities': densities, 'metallicities': metallicities, 'temperatures': temperatures, 'timeArr_in_sec': timeArr_in_sec, 'kpc': kpcsF,
         'uEvolution': uEvolutionX, 'muArr': muArrX, 'metalz': metalzX, 'TEvolution': TEvolutionX}
with open('coolHeatGridNew.pkl', 'wb') as f:
  pickle.dump(dictx, f)


with open('coolHeatGridNewdebug.bin', 'wb') as f:
  # Save N_Kpc, N_nH, N_Z, N_T, N_time for dimensions
  f.write(struct.pack('i', N_kpc))
  f.write(struct.pack('i', N_nH))
  f.write(struct.pack('i', N_Z))
  f.write(struct.pack('i', N_T))
  f.write(struct.pack('i', N_M))
  f.write(struct.pack('i', N_time))
  
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
              
  # Save TEvolution
  for t in range(N_kpc):
    for i in range(N_T):
      for j in range(N_nH):
        for k in range(N_Z):
          for l in range(N_time):
            f.write(struct.pack('f', TEvolutionX[t][i][j][k][l]))  # use 'f' format for float



print('Elapsed time = ', time.time() - TT)





