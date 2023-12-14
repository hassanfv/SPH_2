
import numpy as np
import matplotlib.pyplot as plt
import struct

#filename = './No_T_limit/G-0.002200.bin'

filename = 'G-0.034000.bin'


def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        # Read N and N_ionFrac
        N, N_ionFrac = struct.unpack('ii', f.read(2 * 4))  # 4 bytes each for two integers

        # Read arrays
        Typ = np.array(struct.unpack(f'{N}i', f.read(N * 4)))
        x = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        y = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        z = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vx = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vy = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vz = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        rho = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        h = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        u = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        mass = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        ionFrac = np.array(struct.unpack(f'{N_ionFrac}f', f.read(N_ionFrac * 4)))
        ngbDebug = np.array(struct.unpack(f'{N}i', f.read(N * 4)))  # Reading ngbDebug

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)


n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]

h = h[n]

vx = vx[n]
vy = vy[n]
vz = vz[n]

v = np.sqrt(vx*vx + vy*vy + vz*vz)
print('np.sort(v) = ', np.sort(v))

C = 0.25 # Courant coeff.

print('np.sort(h) = ', np.sort(h))

dt = C * min(h) / max(v)
print()
print('dt = ', dt)




