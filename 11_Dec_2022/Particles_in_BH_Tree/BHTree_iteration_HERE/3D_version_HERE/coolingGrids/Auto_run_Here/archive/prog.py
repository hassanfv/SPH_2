
import os
import shutil

# Read the original file
with open('grid_noneq_evolution_AGN.param', 'r') as file:
    file_content = file.readlines()

# List of new distance values
dist = [0.1, 0.2, 0.3, 0.4, 0.5]

new_files = []

for i in range(len(dist)):
    # Create a new file name based on the distance value
    new_file_name = f'grid_noneq_evolution_{dist[i]}kpc.param'
    
    output_dir = f'./{str(dist[i])}kpc/'
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    new_file_path = os.path.join(output_dir, new_file_name)

    # Create a copy of the original content and update the necessary lines
    new_content = []
    for line in file_content:
        if line.startswith('output_file'):
            new_content.append(f'output_file    {output_dir}grid_noneq_evolution_{dist[i]}kpc.hdf5\n')
        elif line.startswith('distance_to_AGN_kpc'):
            new_content.append(f'distance_to_AGN_kpc {dist[i]}\n')
        else:
            new_content.append(line)

    # Write the new content to the new file
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(new_content)
    
    new_files.append(new_file_path)

new_files  # List of paths to the new files created

