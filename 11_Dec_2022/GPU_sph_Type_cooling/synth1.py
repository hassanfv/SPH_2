import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#filename = 'G-0.001313.bin' # No cooling
#filename = './OutAGN_1.0kpc_New/G-0.000300.bin' # With cooling

#filename = './OutAGN_1.0kpc_res_by_2/G-0.004450.bin'

filename = './Out720k/G-0.021000.bin'


unit_rho = 2.84247273967381e-23
unit_length = 3.086e+21 #cm
unit_u = 18067325774465.332
Unit_time_in_s = 7.260E+14 #seconds


kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.
XH = 0.7

#===== getDensityx
@njit
def getDensity(r, pos, m, h):  # r is the position of all particles. pos is the positions for which we want to know rho !

  N = r.shape[0]
  M = pos.shape[0]

  rho = np.zeros(M)

  for i in range(M):

    s = 0.0

    for j in range(N):

      dx = pos[i, 0] - r[j, 0]
      dy = pos[i, 1] - r[j, 1]
      dz = pos[i, 2] - r[j, 2]
      rr = (dx**2 + dy**2 + dz**2)**0.5

      hij = 0.5 * (h[i] + h[j])

      sig = 1.0/np.pi
      q = rr / hij

      WIij = 0.0

      if q <= 1.0:
        WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

      if (q > 1.0) and (q <= 2.0):
        WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
        
      s += m[j] * WIij

    rho[i] = s

  return rho


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

    return N, Typ, x, y, z, vx, vy, vz, rho, h, u, uBAd, uAC, mass, dudt, utprevious

# Usage
N, Typ, x, y, z, vx, vy, vz, rho, h, u, uBAd, uAC, mass, dudt, utprevious = loadArraysFromBinary(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]
#Temp = (gamma - 1) * mH / kB * mu * u * unit_u

x = x[n]
y = y[n]
z = z[n]

h = h[n]

mass = mass[n]

r = np.vstack((x, y, z))
r = np.transpose(r)

print(r.shape)

dl = 0.01

grid = np.arange(0., 0.6, dl)

Ngrid = len(grid)

grid_vec = np.zeros((Ngrid, 3))

grid_vec[:, 0] = grid

rho_cgs = getDensity(r, grid_vec, mass, h) * unit_rho
nH_cgs = rho_cgs * XH / mH
print('nH_cgs = ', nH_cgs)
print()

ni_nH = 0.1 # ratio of HI to Htot

nHI = ni_nH * nH_cgs

dl_cgs = dl * unit_length

Ncol = nHI * dl_cgs

print(Ncol)

unit_u = 1.032419E+12

u_cgs = u * unit_u

print()
print('sort(u_cgs) = ', np.sort(u_cgs))
print()

u_t = 1e12

#--- test values ---
r_p = 0.42 # kpc
u_p = u_t
nH_p = 1000.
Z_p = -1.0
dt_sec = 3.630E+08 # seconds
#-------------------

from abundance_finder2 import hcooler
Abund = hcooler(r_p, u_p, nH_p, Z_p, dt_sec)

print()
print('Abund = ', Abund)
print()

print('Abund[1] = ', Abund[1])
print('Abund[2] = ', Abund[2])





