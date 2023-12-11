import struct
import matplotlib.pyplot as plt


def read_vectors_from_file(filename):
    with open(filename, 'rb') as file:
        # Read the size of the vectors
        N = struct.unpack('i', file.read(4))[0]

        # Function to read a vector of a specific type and size
        def read_vector(fmt, size):
            return struct.unpack(f'{size}{fmt}', file.read(4 * size))

        # Read the vectors
        Typvec = read_vector('i', N)
        xvec = read_vector('f', N)
        yvec = read_vector('f', N)
        zvec = read_vector('f', N)
        vxvec = read_vector('f', N)
        vyvec = read_vector('f', N)
        vzvec = read_vector('f', N)
        uvec = read_vector('f', N)
        hvec = read_vector('f', N)
        epsvec = read_vector('f', N)
        massvec = read_vector('f', N)

    return Typvec, xvec, yvec, zvec, vxvec, vyvec, vzvec, uvec, hvec, epsvec, massvec

# Usage
filename = 'IC_R_1334k.bin'
data = read_vectors_from_file(filename)

Typvec, x, y, z, vxvec, vyvec, vzvec, uvec, hvec, epsvec, massvec = data

plt.scatter(x, y, s = 0.1)

plt.show()



