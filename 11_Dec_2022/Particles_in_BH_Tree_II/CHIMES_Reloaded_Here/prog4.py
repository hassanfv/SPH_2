
import numpy as np
import pandas as pd
import pickle
import readchar
import os
import time
import h5py
from mpi4py import MPI
import subprocess


gamma = 5./3.
kB = 1.3807e-16
mH = 1.6726e-24

#----- Temp_to_u
def Temp_to_u(T, Ab):

  s = 0.0
  p = 0.0
  for j in range(157):
    s += Ab[j] * AtomicMass[j]
    p += Ab[j] # Note that ne is also included in the sum!!

  mu = s / p

  utmp = kB / mH / (gamma - 1.) / mu * T
  
  return utmp, np.round(mu, 4)


#===== mainFunc
def mainFunc(nbeg, nend):

  Tres = []
  AbRes = []
  uRes = []
  muRes = []

  for i in range(nbeg, nend):

    lst = inLists[i]

    OutFile = './TempFiles/' + 'grid_' + str(i) + '.hdf5'

    L_max = 3.086e23
    Lsh = 10**lst[3] / 10**lst[1]

    Lsh = min(Lsh, L_max)
    

    user_input = {
      "output_file": "   " + OutFile + '\n',
      "distance_to_AGN_kpc": f'{lst[2]:.3f}\n',
      "log_T_min": "     " + f'{lst[0]:.3f}\n',
      "log_T_max": "     " + f'{lst[0]:.3f}\n',
      "log_nH_min": "   " +  f'{lst[1]:.3f}\n',
      "log_nH_max": "   " +  f'{lst[1]:.3f}\n',
      "max_shield_length": " " + f'{Lsh:.3E}\n'}

    # Update the parameters in the file content
    updated_content = update_parameters(original_content, user_input)

    # Write the updated content to a new file
    updated_file_path = './TempFiles/' + 'grid_' + str(i) + '.param'
    with open(updated_file_path, 'w') as file:
        file.writelines(updated_content)

    #---- Executing CHIMES ----
    command = f"python3 chimes-driverx.py {updated_file_path}"
    os.system(command)
    #----

    #--------> Reading the hdf5 file and collecting the data <---------
    f = h5py.File(OutFile, 'r')
    
    TEvolution = f['TemperatureEvolution'][:]
    TEvol = list(TEvolution[0, 0, 0, :])

    #--------> Handling abundances <----------
    AbundEvolution = f['AbundanceEvolution'][:]

    for k in range(N_time):
      Ab = AbundEvolution[0, 0, 0, :, k]
      utmp, mu = Temp_to_u(TEvol[k], Ab)
      uRes += [utmp]
      muRes += [mu]

    nxIDz = [1,   2,     7,     8,      9,     10,    58,     59,      60,    19,   23,  28,    73,    111,    44,    45]
    IDz = ['HI', 'HII', 'CI', 'CII', 'CIII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OI' 'OVI', 'SII', 'FeII', 'MgI', 'MgII']
    N_IDz = len(IDz)
    
    for j in range(N_IDz):
      for k in range(N_time):
        AbundEvol = AbundEvolution[0, 0, 0, nxIDz[j], k] # order ---> [T, nH, r, NH, Elm, time]
        if AbundEvol < 1e-30:
          AbundEvol = 0.0
        AbRes.append(AbundEvol)

    Tres += TEvol
    

    os.remove(OutFile)
    os.remove(updated_file_path)

  return Tres, uRes, muRes, AbRes



#===== update_parameters
def update_parameters(content, updates):
    
    for i, line in enumerate(content):
        # Split the line into components assuming space or tab delimitation
        parts = line.split()
        # Check if the line contains a parameter that needs to be updated
        if len(parts) > 1 and parts[0] in updates:
            # Update the parameter value based on the user input
            parts[1] = str(updates[parts[0]])
            # Reconstruct the line with the updated value
            content[i] = ' '.join(parts)
    return content




df = pd.read_csv('data_species.csv')
print(df)
AtomicMass = df['A']

# Read the initial param file.
with open('grid_noneq_evolution_AGN.param', 'r') as file:
    original_content = file.readlines()


with open('inputListsX.pkl', 'rb') as f:
  inLists = pickle.load(f)

N_time = 11 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Double Check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

N = 20#len(inLists)

#------- used in MPI --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#--------------------------

if rank == 0:
	T1 = time.time()
#--------------------------

local_res = mainFunc(nbeg, nend)
Xres1 = 0 # This is just a placeholder. Its absence would crash this line--> Xres = comm.bcast(Xres, root = 0) for CPUs with rank != 0.
Xres2 = 0
Xres3 = 0
Xres4 = 0


if rank == 0:
	Xres1, Xres2, Xres3, Xres4 = local_res
	for i in range(1, nCPUs):
		res_tmp1, res_tmp2, res_tmp3, res_tmp4 = comm.recv(source = i)
		Xres1 += res_tmp1
		Xres2 += res_tmp2
		Xres3 += res_tmp3
		Xres4 += res_tmp4
else:
	comm.send(local_res, dest = 0)

Xres1 = comm.bcast(Xres1, root = 0) # Tres
Xres2 = comm.bcast(Xres2, root = 0) # uRes
Xres3 = comm.bcast(Xres3, root = 0) # muRes
Xres4 = comm.bcast(Xres4, root = 0) # AbRes
#----------------------------

if rank == 0:
  print()
  print(Xres4)
  
  # WRITE as binary file so that we can read it in C++!!!!

#if rank == 0:
#  print(f'Elapsed time = ', time.time() - T1)




