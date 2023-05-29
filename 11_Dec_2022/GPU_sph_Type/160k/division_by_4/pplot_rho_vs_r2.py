
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

filename = 'G-0.000140.bin'

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

print(np.sum(Typ == -1))

nT = np.where(Typ != -1)[0]
x = x[nT]
y = y[nT]
z = z[nT]
rho = rho[nT]

nz = np.where(np.abs(z) < 0.040)[0]

x = x[nz]
y = y[nz]
z = z[nz]
rho = rho[nz]

rr = (x*x + y*y + z*z)**0.5

print(min(rr), max(rr))

rgrid = np.linspace(min(rr), max(rr), 500)

res = []

for i in range(0, len(rgrid)-1):
  
  nn = np.where((rr >= rgrid[i]) & (rr < rgrid[i+1]))[0]
  
  res.append([rgrid[i], np.median(rho[nn])])


res = np.array(res)
dd = res[:, 0]
rho = res[:, 1]

#srho = gaussian_filter1d(srho, sigma=10)

plt.figure(figsize = (6, 6))
plt.scatter(dd,  rho, s = 5, color = 'k')
#plt.scatter(rgrid, srho, s = 0.1, color = 'r')

xy = 0.22

#plt.xlim(-xy, xy)
#plt.ylim(-xy, xy)

#plt.xlim(0, xy)
#plt.ylim(0, xy)

plt.savefig('fig_rho_vs_r.png')

plt.show()









