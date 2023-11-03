import struct
import matplotlib.pyplot as plt
import numpy as np


filename = "data3d.bin"

with open(filename, "rb") as file:
    # Read N and GridSize
    N, GridSize = struct.unpack('ii', file.read(8)) # 8 because each int is 4 bytes

    # Read the x, y, and z arrays
    x = np.array(struct.unpack('f' * N, file.read(N * 4)))
    y = np.array(struct.unpack('f' * N, file.read(N * 4)))
    z = np.array(struct.unpack('f' * N, file.read(N * 4)))

    # Read cell_particles_offsets and cell_particles_values
    cell_particles_offsets = list(struct.unpack('i' * (GridSize * GridSize * GridSize + 1), file.read((GridSize * GridSize * GridSize + 1) * 4)))
    cell_particles_values = list(struct.unpack('i' * N, file.read(N * 4)))

    # Read neighbors
    neighbors_size, = struct.unpack('i', file.read(4))
    neighbors = list(struct.unpack('i' * neighbors_size, file.read(neighbors_size * 4)))




plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[neighbors], y[neighbors], s = 5, color = 'lime')

plt.show()



