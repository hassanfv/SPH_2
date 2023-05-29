
import numpy as np
import matplotlib.pyplot as plt

filename = 'G-0.000320.bin'

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


nz = np.where(np.abs(z) < 0.04)[0]


plt.figure(figsize = (6, 6))
plt.scatter(x[nz], y[nz], s = 0.01, color = 'k')
plt.scatter([0, 0], [0, 0], s = 20, color = 'r')

xy = 0.28

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

#plt.xlim(0, xy)
#plt.ylim(0, xy)

plt.savefig('fig.png')

plt.show()









