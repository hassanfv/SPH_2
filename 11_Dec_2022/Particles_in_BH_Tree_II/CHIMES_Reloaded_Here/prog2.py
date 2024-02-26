
import numpy as np
import pandas as pd
import pickle
import readchar
import os
import time
import h5py


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


# Read the initial param file.
with open('grid_noneq_evolution_AGN.param', 'r') as file:
    original_content = file.readlines()


with open('inputListsX.pkl', 'rb') as f:
  inLists = pickle.load(f)


nbeg = 0
nend = 10

T1 = time.time()

Tres = []

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
  
  print(TEvolution.shape)
  print(type(TEvolution))
  print(TEvol)
  print(len(TEvol))
  print(type(TEvol))
  
  Tres += TEvol
  
  print(OutFile)
  os.remove(OutFile)
  os.remove(updated_file_path)
  

with open('data.pkl', 'wb') as f:
  pickle.dump(Tres, f)



print(f'Elapsed time = ', time.time() - T1)




