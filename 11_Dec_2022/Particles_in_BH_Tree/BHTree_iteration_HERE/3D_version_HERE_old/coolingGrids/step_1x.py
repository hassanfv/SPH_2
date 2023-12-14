
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

N = 51 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE ACCORDINGLY !!!!! N is the number of values in the time array!!!!!!!!

#------- used in MPI --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#----------------------------

gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

df = pd.read_csv('data_species.csv')
print(df)
    
AtomicMass = df['A']

dist = '0.1' # kpc

f = h5py.File('./coolHeatGrids_hdf5/grid_noneq_evolution_' + dist + 'kpc.hdf5', 'r')

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

TT = time.time()



def h_func(N_T, N_nH, N_Z, nbeg, nend):

    M = nend - nbeg
    
    uEvolution = np.zeros((N_T, N_nH, N_Z, M))
    muArr = np.zeros((N_T, N_nH, N_Z, M))
    
    metalz = np.zeros((N_T, N_nH, N_Z, 14, M))

    for i in range(nbeg, nend):
        for ndx_nH in range(n_densities):
            for ndx_Z in range(n_metallicities):
                for ndx_T in range(n_temperatures):
                
                    T = TemperatureEvolution[ndx_T, ndx_nH, ndx_Z, i]
                    
                    Ab = AbundanceEvolution[ndx_T, ndx_nH, ndx_Z, :, i]
                    s = 0.0
                    p = 0.0
                    for j in range(157):
                        s += Ab[j] * AtomicMass[j]
                        p += Ab[j] # Note that ne is also included in the sum!!

                    mu = s / p

                    #if i == 0: print(f'T = {T},  mu = {mu}')

                    utmp = kB / mH / (gamma - 1.) / mu * T
                    uEvolution[ndx_T, ndx_nH, ndx_Z, i-nbeg] = utmp
                    muArr[ndx_T, ndx_nH, ndx_Z, i-nbeg] = mu
                    
                    #        HI  HII  CI  CII  CII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII
                    nxIDz = [1,  2,   7,  8,   9,   10,  58,   59,    60,   19, 28,  111,  44,  45]
                    IDz = ['HI', 'HII', 'CI', 'CII', 'CII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']
                    for ii in range(len(IDz)):
                      metalz[ndx_T, ndx_nH, ndx_Z, ii, i-nbeg] = Ab[nxIDz[ii]]

    return uEvolution, muArr, metalz


TA = time.time()
#--------- cooler (main) ---------
local_res = h_func(N_T, N_nH, N_Z, nbeg, nend)
res = 0
muA = 0
metalz = 0

if rank == 0:
	res, muA, metalz = local_res
	for i in range(1, nCPUs):
		res_tmp, muA_tmp, metalz_tmp = comm.recv(source = i)
		res = np.concatenate((res, res_tmp), axis = 3)
		muA = np.concatenate((muA, muA_tmp), axis = 3)
		metalz = np.concatenate((metalz, metalz_tmp), axis = 4)
		
else:
	comm.send(local_res, dest = 0)

res = comm.bcast(res, root = 0)
muA = comm.bcast(muA, root = 0)
metalz = comm.bcast(metalz, root = 0)
comm.Barrier()

print('TA = ', time.time() - TA)
#----------------------------

densities = 10**densities
metallicities = 10**metallicities
#temperatures = 10**temperatures

if rank == 0:
	print('Total Elapsed time = ', time.time() - TT)

if rank == 0:
	print(res.shape)
	dictx = {'densities': densities, 'metallicities': metallicities, 'temperatures': temperatures, 'timeArr_in_sec': timeArr,
	         'uEvolution': res, 'muArr': muA, 'metalz': metalz}
	with open('./coolHeatGrids_pkl/coolHeatGrid_' + dist + 'kpc.pkl', 'wb') as f:
		pickle.dump(dictx, f)


f.close()

#print('muA = ', np.sort(muA))

if rank == 0:
    with open('./coolHeatGrids_bin/coolHeatGrid_' + dist + 'kpc.bin', 'wb') as f:
        # Save N_nH, N_Z, N_T, N_time for dimensions
        f.write(struct.pack('i', N_nH))
        f.write(struct.pack('i', N_Z))
        f.write(struct.pack('i', N_T))
        f.write(struct.pack('i', N_time)) # Note that len(muArr) = N_time!
        
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
        for val in timeArr:
            f.write(struct.pack('f', val))  # use 'f' format for float
            
        # Save res
        for i in range(N_T):
            for j in range(N_nH):
                for k in range(N_Z):
                    for l in range(N_time):
                        f.write(struct.pack('f', res[i][j][k][l]))  # use 'f' format for float
    
        # Save muA
        for i in range(N_T):
            for j in range(N_nH):
                for k in range(N_Z):
                    for l in range(N_time):
                        f.write(struct.pack('f', muA[i][j][k][l]))  # use 'f' format for float
        # Save metalz
        for i in range(N_T):
            for j in range(N_nH):
                for k in range(N_Z):
                    for ii in range(14):
                      for l in range(N_time):
                          f.write(struct.pack('f', metalz[i][j][k][ii][l]))  # use 'f' format for float



