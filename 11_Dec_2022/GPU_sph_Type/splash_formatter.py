import numpy as np
import matplotlib.pyplot as plt

filename = 'G-0.000400.bin'

filename = './Outputs_300k_particles/G-0.000680.bin'

def loadArraysFromBinary(filename):
    with open(filename, "rb") as file:
        # Read N
        N = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Create arrays for each of the data types
        Typ = np.fromfile(file, dtype=np.int32, count=N)
        x = np.fromfile(file, dtype=np.float32, count=N)
        y = np.fromfile(file, dtype=np.float32, count=N)
        z = np.fromfile(file, dtype=np.float32, count=N)
        vx = np.fromfile(file, dtype=np.float32, count=N)
        vy = np.fromfile(file, dtype=np.float32, count=N)
        vz = np.fromfile(file, dtype=np.float32, count=N)
        rho = np.fromfile(file, dtype=np.float32, count=N)
        h = np.fromfile(file, dtype=np.float32, count=N)
        u = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)

    return N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass

# Usage
N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass = loadArraysFromBinary(filename)

nx = np.where(u != 0.0)[0]

x = x[nx]
y = y[nx]
z = z[nx]

vx = vx[nx]
vy = vy[nx]
vz = vz[nx]

# Write to file
with open('data.txt', 'w') as f:
    for values in zip(x, y, z, vx, vy, vz):
        f.write(' '.join(map(str, values)) + '\n')






