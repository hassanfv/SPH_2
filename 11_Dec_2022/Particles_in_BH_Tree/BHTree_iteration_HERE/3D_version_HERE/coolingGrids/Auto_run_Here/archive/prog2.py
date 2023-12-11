
import os
import shutil
import numpy as np

# Read the original file
with open('grid_noneq_evolution_AGN.param', 'r') as file:
    file_content = file.readlines()

# List of new distance values
dist = np.arange(0.1, 1.2, 0.1)
print('dist = ', dist) # Note that things like 0.3000000001 will be rounded in the main code when used!

TiMe = np.arange(1, 11)

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
        new_content.append(f'output_file    {output_dir}grid_noneq_evolution_{round(dist[i], 1)}kpc_{TiMe[j]}yrs.hdf5\n')
            
      elif line.startswith('distance_to_AGN_kpc'):
        new_content.append(f'distance_to_AGN_kpc {round(dist[i], 1)}\n')
            
      elif line.startswith('hydro_timestep'):
        new_content.append(f'hydro_timestep            {TiMe[j]:.2E}\n')
        
      else:
        new_content.append(line)

    # Write the new content to the new file
    with open(new_file_path, 'w') as new_file:
      new_file.writelines(new_content)
    



