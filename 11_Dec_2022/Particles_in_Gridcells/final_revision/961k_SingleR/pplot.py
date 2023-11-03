
import numpy as np
import matplotlib.pyplot as plt


filename = 'G-0.000000.bin'


import struct

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

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac 



# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(Typ == 0)[0]
rho = rho[n]
u = u[n]
h = h[n]

x = x[n]
y = y[n]
z = z[n]

nz = np.where(np.abs(z) < 0.04)[0]

x = x[nz]
y = y[nz]
z = z[nz]

h = h[nz]

print(np.sort(h))
print(h.shape)

n0 = np.where(h == 0.0)[0]


plt.figure(figsize = (8, 8))
plt.scatter(x, y, color = 'k', s = 0.1)
plt.scatter(x[n0], y[n0], color = 'r', s = 10)
#---------------------------

xy = 0.40

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')

plt.show()




