
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#filename = 'G-0.000260.bin'
#filename = './OutXX/G-0.000180.bin'

#filename = './OutAGN_1.0kpc_res_by_2/G-0.002500.bin'

filename = './Out720k/G-0.021000.bin'

unit_rho = 2.842E-23
unit_u = 1.8067E+13
unit_time = 7.260E+14

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
        uBAd = np.fromfile(file, dtype=np.float32, count=N)
        uAC = np.fromfile(file, dtype=np.float32, count=N)
        mass = np.fromfile(file, dtype=np.float32, count=N)
        
        dudt = np.fromfile(file, dtype=np.float32, count=N)
        utprevious = np.fromfile(file, dtype=np.float32, count=N)

    #return N, Typ, x, y, z, vx, vy, vz, rho, h, u, uB, mass
    return N, Typ, x, y, z, vx, vy, vz, rho, h, u, uBAd, uAC, mass, dudt, utprevious

# Usage
N, Typ, x, y, z, vx, vy, vz, rho, h, u, uBAd, uAC, mass, dudt, utprevious = loadArraysFromBinary(filename)


print(np.sum(Typ == -1))

nT = np.where(Typ != -1)[0]
x = x[nT]
y = y[nT]
z = z[nT]
rho = rho[nT]

nz = np.where(np.abs(z) < 100.040)[0]

x = x[nz]
y = y[nz]
z = z[nz]
rho = rho[nz]
rhoT = rho.copy()

mH = 1.6726e-24;
XH = 0.70;
muT = 1.0; # For now we assume it to be 1.0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rho_cgs = rho * unit_rho;
#nGas = rho_cgs / (muT * mH);
nH = XH * rho_cgs / mH;

print()
print(f'min(nH) = {min(nH):.3f},  max(nH) = {max(nH):.3f} NOTE: the min and max are before creating grids, that is why they differ from plot!!!')
print()

rr = (x*x + y*y + z*z)**0.5

print(min(rr), max(rr))

#rgrid = np.linspace(min(rr), max(rr), 3000)
rgrid = np.logspace(np.log10(min(rr)), np.log10(max(rr)), 6000)

res = []

for i in range(0, len(rgrid)-1):
  
  nn = np.where((rr >= rgrid[i]) & (rr < rgrid[i+1]))[0]
  
  res.append([rgrid[i], np.median(rho[nn])])


res = np.array(res)
dd = res[:, 0]
rho = res[:, 1]

#srho = gaussian_filter1d(srho, sigma=10)

mH = 1.6726e-24;
XH = 0.70;
muT = 1.0; # For now we assume it to be 1.0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rho_cgs = rho * unit_rho;
#nGas = rho_cgs / (muT * mH);
nH = XH * rho_cgs / mH;

plt.figure(figsize = (6, 6))
#plt.scatter(rr,  rhoT, s = 0.001, color = 'k')
plt.scatter(dd, nH, s = 5, color = 'k')

xy = 0.22

#plt.xlim(-xy, xy)
#plt.ylim(-xy, xy)

#plt.xlim(0, xy)
#plt.ylim(0, xy)

plt.savefig('fig_nH_vs_r.png')

plt.show()


