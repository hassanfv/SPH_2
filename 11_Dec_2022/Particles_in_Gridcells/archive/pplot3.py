import struct
import matplotlib.pyplot as plt
import numpy as np



filename = "data.bin"

with open(filename, "rb") as file:
    # Read N and GridSize
    N = struct.unpack('i', file.read(4))[0]
    GridSize = struct.unpack('i', file.read(4))[0]
    
    # Read x and y arrays
    x = np.array(struct.unpack(f'{N}f', file.read(N * 4)))
    y = np.array(struct.unpack(f'{N}f', file.read(N * 4)))

    # Read cell_particles_offsets
    offsets_size = struct.unpack('i', file.read(4))[0]
    cell_particles_offsets = list(struct.unpack(f'{offsets_size}i', file.read(offsets_size * 4)))
    
    # Read cell_particles_values
    values_size = struct.unpack('i', file.read(4))[0]
    cell_particles_values = list(struct.unpack(f'{values_size}i', file.read(values_size * 4)))



i = 19  # x-axis
j = 7 # y-axis

ndx = i + j * GridSize

nbeg = cell_particles_offsets[ndx]
nend = cell_particles_offsets[ndx+1]

n = cell_particles_values[nbeg:nend]


plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[n], y[n], s = 5, color = 'lime')

plt.show()



