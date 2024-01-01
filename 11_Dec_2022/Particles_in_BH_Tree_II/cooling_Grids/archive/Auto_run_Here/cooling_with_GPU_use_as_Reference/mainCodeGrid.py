
import os
import shutil
import numpy as np
import time
import subprocess

yrs_in_sec = 1.0 * 365.25 * 24.0 * 3600.0
print(f'yrs_in_sec = {yrs_in_sec:.3E}')

# Read a param file for reference. This file should exist in the current directory!
with open('grid_noneq_evolution_AGN.param', 'r') as file:
    file_content = file.readlines()


dist = np.arange(0.1, 1.08, 0.1) # kpc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print('dist = ', dist) # Note that things like 0.3000000001 will be rounded in the main code when used!

TiMe = np.arange(1, 11, 0.5) # yrs !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(TiMe)

#---- Deleting old files and folders ------
for i in range(len(dist)):
  print(round(dist[i], 1))
  output_dir = f'./{round(dist[i], 1)}kpc/'
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
#------------------------------------------

for i in range(len(dist)):
  for j in range(len(TiMe)):
    # Create a new file name based on the distance value
    new_file_name = f'grid_noneq_evolution_{round(dist[i], 1)}kpc_{TiMe[j]}yrs.param'
    
    output_dir = f'./{round(dist[i], 1)}kpc/'
    
    if j == 0: # We need to create the dir only once!!!
      os.makedirs(output_dir)
    
    new_file_path = os.path.join(output_dir, new_file_name)

    # Create a copy of the original content and update the necessary lines
    new_content = []
    for line in file_content:
      if line.startswith('output_file'):
        new_content.append(f'output_file    {output_dir}grid_noneq_evolution_{round(dist[i], 1)}kpc_{str(TiMe[j]).zfill(2)}yrs.hdf5\n')
            
      elif line.startswith('distance_to_AGN_kpc'):
        new_content.append(f'distance_to_AGN_kpc {round(dist[i], 1)}\n')
            
      elif line.startswith('hydro_timestep'):
        new_content.append(f'hydro_timestep            {(yrs_in_sec * TiMe[j] / 10):.2E}\n') # 10 is n_iterations!
        
      else:
        new_content.append(line)

    # Write the new content to the new file
    with open(new_file_path, 'w') as new_file:
      new_file.writelines(new_content)
    
    #-- Now we run CHIMES with this created param file:
    # Command to be executed
    command = f'mpirun -np 96 python3.10 chimes-driver.py {output_dir}grid_noneq_evolution_{round(dist[i], 1)}kpc_{TiMe[j]}yrs.param'
    
    print(command)

    # Using subprocess to run the command
    TA = time.time()
    
    try:
      result = subprocess.run(command, shell=True, check=True)
      print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
      print(f"An error occurred: {e}")

    print('Elapsed time = ', time.time() - TA)
    



