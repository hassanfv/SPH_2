
import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter1d
import struct

filename = 'G-0.029600.bin'

#filename = './No_T_limit/G-0.003600.bin'

unit_rho = 2.03481e-23
unit_u = 1.29337e+13
unit_time = 8.58094e+14

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


print(np.sum(Typ != -1))

n = np.where(u != 0.0)[0]
x = x[n]
y = y[n]
z = z[n]
rho = rho[n]
u = u[n]

nz = np.where(np.abs(z) < 100.040)[0]

x = x[nz]
y = y[nz]
z = z[nz]
rho = rho[nz]
rhoT = rho.copy()
u = u[nz]

gamma = 5./3.
mH = 1.6726e-24;
XH = 0.70;
kB = 1.3807e-16
mu = 0.6; # For now we assume it to be 1.0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rho_cgs = rho * unit_rho;
nH = XH * rho_cgs / mH;

print()
print(f'min(nH) = {min(nH):.3f},  max(nH) = {max(nH):.3f} NOTE: the min and max are before creating grids, that is why they differ from plot!!!')
print()

rr = (x*x + y*y + z*z)**0.5

print(min(rr), max(rr))


Temp = (gamma - 1) * mH / kB * mu * u * unit_u

rgrid = np.logspace(np.log10(min(rr)), np.log10(max(rr)), 6000)

res = []

for i in range(0, len(rgrid)-1):
  
  nn = np.where((rr >= rgrid[i]) & (rr < rgrid[i+1]))[0]
  
  res.append([rgrid[i], np.median(np.log10(Temp[nn]))])


res = np.array(res)
dd = res[:, 0]
Temp = res[:, 1]


plt.figure(figsize = (6, 6))
plt.scatter(dd, Temp, s = 5, color = 'k')

plt.savefig('fig_Temp_vs_r.png')

plt.show()


