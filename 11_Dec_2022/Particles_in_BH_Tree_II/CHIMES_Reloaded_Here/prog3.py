
import numpy as np
import pandas as pd
import pickle
import readchar
import os
import time
import h5py


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


nbeg = 0
nend = 20

N_time = 11 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Double Check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

T1 = time.time()

Tres = []
AbRes = []
uRes = []
muRes = []

for i in range(nbeg, nend):

  lst = inLists[i]
  print(lst)

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
  command = f"python3 chimes-driver.py {updated_file_path}"
  os.system(command)
  #----

  #--------> Reading the hdf5 file and collecting the data <---------
  f = h5py.File(OutFile, 'r')
  
  TEvolution = f['TemperatureEvolution'][:]
  TEvol = list(TEvolution[0, 0, 0, :])

  #--------> Handling abundances <----------
  AbundEvolution = f['AbundanceEvolution'][:]
  print(AbundEvolution.shape)

  for k in range(N_time):
    Ab = AbundEvolution[0, 0, 0, :, k]
    utmp, mu = Temp_to_u(TEvol[k], Ab)
    #uRes += [format(utmp, '.5e')]
    #muRes += [format(mu, '.4f')]
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
      #AbRes.append(format(AbundEvol, '.4e'))
      AbRes.append(AbundEvol)
      
  
  #TEvol = [format(_, '.3f') for _ in TEvol]
  Tres += TEvol
  
  print(TEvol) 
  print()
  print(uRes)
  print()
  print(muRes)
  print()
  print(Tres)
  print()
  print(AbRes)
  

  print(OutFile)
  os.remove(OutFile)
  os.remove(updated_file_path)


with open('data.pkl', 'wb') as f: # Use post1.py to read this file!
  pickle.dump(AbRes, f)



print(f'Elapsed time = ', time.time() - T1)




