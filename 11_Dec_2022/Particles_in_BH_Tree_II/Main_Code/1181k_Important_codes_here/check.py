import numpy as np
import matplotlib.pyplot as plt
from numba import njit


filename = './Outputs/G-0.017600.bin'

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!
unit_length = 3.086e+21 #!!!!!!!!!!!!!!!!!!!

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

nt = np.where(nH_cgs == max(nH_cgs))[0][0] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print('max nH coords = ', (x[nt], y[nt], z[nt]))
print('max nH = ', nH_cgs[nt])
print()

rtmp = np.sqrt((x - x[nt])**2 + (y - y[nt])**2 + (z - z[nt])**2)
ntmp = np.argsort(rtmp)
#ntmp = ntmp[::-1]


#--------- drawing the line of sight -------------
p0 = np.array([0, 0, 0])
p1 = np.array([x[nt], y[nt], z[nt]])

tt = np.linspace(0, 2, 2000)

xt = p0[0] + (p1[0] - p0[0]) * tt
yt = p0[1] + (p1[1] - p0[1]) * tt
zt = p0[2] + (p1[2] - p0[2]) * tt

pt = np.vstack((xt, yt, zt))
pt = np.transpose(pt)
print(pt.shape)

rrt = np.sqrt(pt[:, 0] * pt[:, 0] + pt[:, 1] * pt[:, 1] + pt[:, 2] * pt[:, 2])
#-------------------------------------------------

#----
plt.figure(figsize = (9, 9))

plt.scatter(x, y, s = 0.001, color = 'k')
plt.plot(xt, yt, color = 'red') #----- Line of Sight ---------
xyran = 0.32
plt.xlim(-xyran, xyran)
plt.ylim(-xyran, xyran)
plt.show()
#----


#----
plt.scatter(x[ntmp[:1000]], y[ntmp[:1000]], s = 1, color = 'k')
plt.plot(xt, yt, color = 'red') #----- Line of Sight ---------
delta = 0.02
plt.xlim(x[nt] - delta, x[nt]+delta)
plt.ylim(y[nt] - delta, y[nt]+delta)
plt.show()
#----

#---- plot --> nH_cgs vs T
plt.scatter(nH_cgs[ntmp[:1000]], np.log10(Temp[ntmp[:1000]]), s = 0.5)
plt.show()



n_ion = 0 # HI
iFrac = getIonFrac(r, pt, rho, mass, h, ionFrac[:, n_ion])
T_pt = getTemp(r, pt, rho, Temp, mass, h)

#----- plot ------
#plt.scatter(rrt, nH_pt, s = 5)
plt.scatter(rrt, np.log10(T_pt), s = 5, label = 'T vs. r')
#plt.ylim(3, 11)
plt.legend()
plt.show()


plt.scatter(iFrac, T_pt, s = 5, color = 'k', label = 'iFrac vs. T')
plt.xscale('log')
plt.yscale('log')
#plt.ylim(3, 11)
plt.legend()
plt.show()



#----- plot ------
plt.scatter(rrt, iFrac, s = 5, color = 'k', label = 'iFrac vs. r')
plt.yscale('log')
plt.legend()
plt.show()


rho_cgs = getDensity(r, pt, mass, h) * unit_rho
nH_pt = rho_cgs * XH / mH
Ncol = 0.0

for i in range(len(rrt)-1):

  dl = (rrt[i+1] - rrt[i]) * unit_length
  
  Ncol += nH_pt[i] * dl * iFrac[i]

print('log(N) = ', np.log10(Ncol))

print()



if False:
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






