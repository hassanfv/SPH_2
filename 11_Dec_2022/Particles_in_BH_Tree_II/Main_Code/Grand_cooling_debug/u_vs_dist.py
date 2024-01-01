
import numpy as np
import matplotlib.pyplot as plt
import struct


#filename = './Outputs/G-0.002860.bin'

filename = './Outputs/G-0.000040.bin'

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as file:
        # Read the total number of elements (N)
        N = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Read arrays in the order they were written
        Typ = np.fromfile(file, dtype=np.int32, count=N)
        x = np.fromfile(file, dtype=np.float32, count=N)
        y = np.fromfile(file, dtype=np.float32, count=N)
        z = np.fromfile(file, dtype=np.float32, count=N)
        vx = np.fromfile(file, dtype=np.float32, count=N)
        vy = np.fromfile(file, dtype=np.float32, count=N)
        vz = np.fromfile(file, dtype=np.float32, count=N)
        rho = np.fromfile(file, dtype=np.float32, count=N)
        h = np.fromfile(file, dtype=np.float32, count=N)
        uB = np.fromfile(file, dtype=np.float32, count=N)
        uA = np.fromfile(file, dtype=np.float32, count=N)
        u = np.fromfile(file, dtype=np.float32, count=N)
        dudt = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)

    return x, y, z, vx, vy, vz, rho, h, uB, uA, u, dudt, mass, Typ, N


# Usage
x, y, z, vx, vy, vz, rho, h, uB, uA, u, dudt, mass, Typ, N = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]

u = u[n]

x = x[n]
y = y[n]
z = z[n]

dist = np.sqrt(x*x + y*y + z*z)

plt.scatter(dist, u, s =0.1)
plt.yscale('log')

plt.show()




