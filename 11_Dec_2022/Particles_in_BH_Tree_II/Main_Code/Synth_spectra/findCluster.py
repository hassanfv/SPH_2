import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from itertools import combinations

filename = '../Out456k/G-0.023500.bin'

unit_u = 4100904397311.213  # !!!!!!!!!!!!!!!!!!!
unit_rho = 6.451817553342665e-24 # !!!!!!!!!!!!!!!!!!!
unit_length = 3.086e+21 #cm # !!!!!!!!!!!!!!!!!!!

XH = 0.7

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.


#===== distance
def distance(i, j):
    return np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)

#===== is_valid_combination
def is_valid_combination(indices):
    for p1 in range(len(indices)):
        for p2 in range(p1+1, len(indices)):
            if distance(indices[p1], indices[p2]) < 0.10:
                return False
    return True


#===== getDensityx
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



#===== getTemp
@njit
def getTemp(r, pos, rho, T, m, h):  # r is the position of all particles. pos is the positions for which we want to know rho !

  N = r.shape[0]
  M = pos.shape[0]

  Temp = np.zeros(M)

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
        
      s += m[j] * T[j] / rho[j] * WIij

    Temp[i] = s

  return Temp


#===== getIonFrac
@njit
def getIonFrac(r, pos, rho, m, h, ionFracx):  # r is the position of all particles. pos is the positions for which we want to know rho !

  N = r.shape[0]
  M = pos.shape[0]

  iFc = np.zeros(M) # iFc = ionFraction at the pt positions!

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
        
      s += m[j] * ionFracx[j] / rho[j] * WIij

    iFc[i] = s

  return iFc # This is the ionFrac for one species (indicated by n_ion)!



#===== readBinaryFile
def readBinaryFile(filename):
    with open(filename, "rb") as file:
        # Read N and N_ionFrac
        N, N_ionFrac = np.fromfile(file, dtype=np.int32, count=2)

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
        ionFrac = np.fromfile(file, dtype=np.float32, count=N_ionFrac)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]
#n = n[::5]
rho = rho[n]
u = u[n]

ionFrac = ionFrac.reshape((N, 14))
ionFrac = ionFrac[n, :]

print(ionFrac.shape)

h = h[n]
mass = mass[n]

x = x[n]
y = y[n]
z = z[n]

#r = np.vstack((x, y, z))
#r = np.transpose(r)
#print(r.shape)

#dist = np.sqrt(x*x + y*y + z*z)

rho_cgs = rho * unit_rho
nH_cgs = rho_cgs * XH / mH

#nx = np.where(nH_cgs > 700)
#x = x[nx]
#y = y[nx]
#z = z[nx]
#print('x.shape = ', x.shape)
#nH_cgs = nH_cgs[nx]

valid_particles = [i for i in range(len(nH_cgs)) if nH_cgs[i] > 900]

print(len(valid_particles))

# If you have less than 4 valid particles, it's impossible to select 4
if len(valid_particles) < 5:
    print("Not enough valid particles.")
else:
    # Check combinations of 4 particles
    for combo in combinations(valid_particles, 3):
        
        if is_valid_combination(combo):
            print(f"Valid set of particles: {combo}")
            combo = list(combo)
            print(print('nH_cgs[combo] = ', nH_cgs[combo]))
            plt.scatter(x, y, s = 1, color = 'k')
            plt.scatter(x[combo], y[combo], s = 30, color = 'r')
            plt.xlim(-0.34, 0.34)
            plt.ylim(-0.34, 0.34)
            plt.show()
            break








