import struct
import matplotlib.pyplot as plt
import numpy as np



with open('data.bin', 'rb') as file:
    # Read i_p
    i_p = struct.unpack('i', file.read(4))[0]
    
    # Read x data
    N = struct.unpack('i', file.read(4))[0]
    x = np.array(struct.unpack(f'{N}f', file.read(4 * N)))
    
    # Read y data
    N = struct.unpack('i', file.read(4))[0]
    y = np.array(struct.unpack(f'{N}f', file.read(4 * N)))
    
    # Read neighboring_particles data
    n_size = struct.unpack('i', file.read(4))[0]
    # Note that n_ngb contains the particles of the neighboring cells and it does not contain other particles
    # of the cell in which i_p particle resides. So be ware of this as when you plot you may get confused as to
    # why the central particles are not colored green !!!!
    n_ngb = list(struct.unpack(f'{n_size}i', file.read(4 * n_size))) # neighboring_particles indices!

#print(i_p, x, y, neighboring_particles)

print('N ngb = ', len(n_ngb))

plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[i_p], y[i_p], s = 5, color = 'r')
plt.scatter(x[n_ngb], y[n_ngb], s = 5, color = 'lime')

plt.show()



