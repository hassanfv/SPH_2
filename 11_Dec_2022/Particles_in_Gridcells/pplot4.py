import struct
import matplotlib.pyplot as plt
import numpy as np

filename = "data.bin"

with open(filename, 'rb') as file:
    # Read N and GridSize
    N = struct.unpack('i', file.read(4))[0]
    GridSize = struct.unpack('i', file.read(4))[0]
    
    # Read x and y arrays
    x = np.array(struct.unpack(f'{N}f', file.read(4 * N)))
    y = np.array(struct.unpack(f'{N}f', file.read(4 * N)))
    
    # Read neighbors
    neighbors_size = struct.unpack('i', file.read(4))[0]
    neighbors = list(struct.unpack(f'{neighbors_size}i', file.read(4 * neighbors_size)))




plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[neighbors], y[neighbors], s = 5, color = 'lime')

plt.show()



