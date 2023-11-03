import numpy as np
import matplotlib.pyplot as plt
from numba import njit


filename = '../Out918k/G-0.103601.bin'

unit_u = 1.5272E+13  # !!!!!!!!!!!!!!!!!!!
unit_rho = 2.403E-23 # !!!!!!!!!!!!!!!!!!!
unit_length = 3.086e+21 #cm # !!!!!!!!!!!!!!!!!!!

XH = 0.7

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.


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

print('Typ == 0 ===> ', np.sum(Typ == 0))

n = np.where(u != 0.0)[0]
rho = rho[n]
u = u[n]

h = h[n]
mass = mass[n]

x = x[n]
y = y[n]
z = z[n]

r = np.vstack((x, y, z))
r = np.transpose(r)
print(r.shape)

vx = vx[n]
vy = vy[n]
vz = vz[n]

rho_cgs = rho * unit_rho
nH_cgs = rho_cgs * XH / mH

Temp = (gamma - 1) * mH / kB * mu * u * unit_u
print('sort T = ', np.sort(Temp))
print('median(Temp) = ', np.median(Temp))


nt = np.where(nH_cgs == max(nH_cgs))[0][0]
print('max nH coords = ', (x[nt], y[nt], z[nt]))

#--------- drawing the line of sight -------------
p0 = np.array([0, 0, 0])
p1 = np.array([x[nt], y[nt], z[nt]])

tt = np.linspace(0, 2, 1000)

xt = p0[0] + (p1[0] - p0[0]) * tt
yt = p0[1] + (p1[1] - p0[1]) * tt
zt = p0[2] + (p1[2] - p0[2]) * tt

pt = np.vstack((xt, yt, zt))
pt = np.transpose(pt)
print(pt.shape)

rrt = np.sqrt(pt[:, 0] * pt[:, 0] + pt[:, 1] * pt[:, 1] + pt[:, 2] * pt[:, 2])
#-------------------------------------------------

rho_cgs = getDensity(r, pt, mass, h) * unit_rho
nH_pt = rho_cgs * XH / mH
Ncol = 0.0

for i in range(len(rrt)-1):

  dl = (rrt[i+1] - rrt[i]) * unit_length
  
  Ncol += nH_pt[i] * dl

print('log(N) = ', np.log10(Ncol))


T_pt = getTemp(r, pt, rho, Temp, mass, h)


#plt.scatter(rrt, nH_pt, s = 5)
plt.scatter(rrt, np.log10(T_pt), s = 5)

plt.axvline(x = 0.39, linestyle = '--', color = 'b')
plt.axvline(x = 0.41, linestyle = '--', color = 'b')

plt.xlim(0.35, 0.45)
plt.show()

s()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=np.log10(nH_cgs), cmap='rainbow', s=2)

#scatter = plt.scatter(x - x[nt], y - y[nt], c=np.log10(nH_cgs), cmap='rainbow', s=2)

plt.colorbar(scatter, label='nH Value')

xy = 0.56
plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig.png')
plt.show()






