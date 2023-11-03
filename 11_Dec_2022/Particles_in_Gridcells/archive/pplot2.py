import struct
import matplotlib.pyplot as plt
import numpy as np



import struct

with open('data.bin', 'rb') as f:
    # Read N
    N = struct.unpack('i', f.read(4))[0]

    # Read i_p
    i_p = struct.unpack('i', f.read(4))[0]

    # Read n_ngb
    size_n_ngb = struct.unpack('i', f.read(4))[0]
    n_ngb = list(struct.unpack(f'{size_n_ngb}i', f.read(4 * size_n_ngb)))

    # Read x and y arrays
    x = np.array(struct.unpack(f'{N}f', f.read(4 * N)))
    y = np.array(struct.unpack(f'{N}f', f.read(4 * N)))


#print(i_p, x, y, neighboring_particles)

print('N ngb = ', len(n_ngb))

plt.figure(figsize = (7, 7))
plt.scatter(x, y, s = 2, color = 'k')
plt.scatter(x[n_ngb], y[n_ngb], s = 5, color = 'lime')
plt.scatter(x[i_p], y[i_p], s = 5, color = 'r')

plt.show()



