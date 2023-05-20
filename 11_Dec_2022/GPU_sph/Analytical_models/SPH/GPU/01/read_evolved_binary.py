
import struct
import matplotlib.pyplot as plt

def read_arrays_from_binary(filename):
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = struct.unpack('i', file.read(4))[0]
        NG = struct.unpack('i', file.read(4))[0]

        # Read the arrays from the file
        x = struct.unpack(f'{N}f', file.read(4 * N))
        y = struct.unpack(f'{N}f', file.read(4 * N))
        z = struct.unpack(f'{N}f', file.read(4 * N))
        vx = struct.unpack(f'{N}f', file.read(4 * N))
        vy = struct.unpack(f'{N}f', file.read(4 * N))
        vz = struct.unpack(f'{N}f', file.read(4 * N))
        rho = struct.unpack(f'{NG}f', file.read(4 * NG))
        h = struct.unpack(f'{NG}f', file.read(4 * NG))
        u = struct.unpack(f'{NG}f', file.read(4 * NG))

    return x, y, z, vx, vy, vz, rho, h, u

# Usage example
filename = './Outputs_1.7Gyr_evolved/G-1.760180.bin'
x, y, z, vx, vy, vz, rho, h, u = read_arrays_from_binary(filename)


plt.scatter(x, y, s = 0.1, color = 'k')

plt.show()
