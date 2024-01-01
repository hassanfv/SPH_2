
import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle

filename = 'G-0.015044.bin'

kB = 1.3807e-16
mu = 0.61 # !!!!!!!!!!!!!!!!!!!!!!!!!! You may want to calculate better mu using ionFrac or ...!
mH = 1.673534e-24
gamma = 5./3.

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the entire file at once

    # Use a memoryview to avoid copies
    buffer = memoryview(file_content)

    # Unpack N and N_ionFrac
    N, N_ionFrac = struct.unpack_from('ii', buffer)
    offset = 8  # Start after the first two integers

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

    # Read arrays
    Typ = read_array(np.int32, N, 4)
    x = read_array(np.float32, N, 4)
    y = read_array(np.float32, N, 4)
    z = read_array(np.float32, N, 4)
    vx = read_array(np.float32, N, 4)
    vy = read_array(np.float32, N, 4)
    vz = read_array(np.float32, N, 4)
    rho = read_array(np.float32, N, 4)
    h = read_array(np.float32, N, 4)
    u = read_array(np.float32, N, 4)
    mass = read_array(np.float32, N, 4)
    ionFrac = read_array(np.float32, N_ionFrac, 4)
    ngbDebug = read_array(np.int32, N, 4)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug

N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)


print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]

rho = rho[n]
u = u[n]
h = h[n]
print(np.sort(h))

x = x[n]
y = y[n]
z = z[n]

rho_cgs = rho * unit_rho
XH = 0.7
nH = rho_cgs * XH / mH

T = (gamma - 1) * mH / kB * mu * u * unit_u

T_crit = 10000 # K
nH_crit = 500

nx = np.where((T < T_crit) & (nH > nH_crit))[0]

print()
print(f'We found {len(nx)} particles matching the criteria!')
print()


nmax_nH = np.where(nH == max(nH))[0]
print()
print(f'maximun nH occurs for particle {nmax_nH}')
print()


with open('clumps.pkl', 'rb') as f:
  clumps = pickle.load(f)


nclump = np.array(list(clumps))
print(nclump)
print(len(nclump))


plt.figure(figsize = (8, 8))

plt.scatter(x, y, s = 0.01, color = 'k')
#plt.scatter(x[nx], y[nx], s = 5.0, color = 'r')
plt.scatter(x[nclump], y[nclump], s = 5.0, color = 'r')

plt.show()







