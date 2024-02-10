import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle



filename = './Outputs/G-0.049590.bin'

unit_velocity_cgs = 1.63333e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 2.66778e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 4.19712e-24 # !!!!!!!!!!!!!!!!!!!
unit_length = 3.086e+21 #!!!!!!!!!!!!!!!!!!!

XH = 0.7

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.


#===== getDensityx
@njit
def getDensity(Typ, r, pos, m, h, h_pt):  # r is the position of all particles. pos is the positions for which we want to know rho !

  N = r.shape[0]
  M = pos.shape[0]

  rho = np.zeros(M)

  for i in range(M):

    s = 0.0

    for j in range(N):

      if Typ[j] == 0:
        dx = pos[i, 0] - r[j, 0]
        dy = pos[i, 1] - r[j, 1]
        dz = pos[i, 2] - r[j, 2]
        rr = (dx**2 + dy**2 + dz**2)**0.5

        hij = 0.5 * (h_pt[i] + h[j])

        sig = 1.0/np.pi
        q = rr / hij

        WIij = 0.0

        if q <= 1.0:
          WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

        if (q > 1.0) and (q <= 2.0):
          WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
          
        s += m[j] * WIij

    rho[i] = s + 1e-15;

  return rho



#===== getIonFrac
@njit
def getIonFrac(Typ, r, pos, rho, m, h, h_pt, ionFracx):  # r is the position of all particles. pos is the positions for which we want to know rho !

  N = r.shape[0]
  M = pos.shape[0]

  iFc = np.zeros((M, 14)) # iFc = ionFraction at the pt positions! pt contains the positions along the line of sight!

  for i in range(M):

    for k in range(14): # k represents the 14 species we have in our ionFrac.

      s = 0.0
      ss = 0.0

      for j in range(N):

        if Typ[j] == 0:
          dx = pos[i, 0] - r[j, 0]
          dy = pos[i, 1] - r[j, 1]
          dz = pos[i, 2] - r[j, 2]
          rr = (dx**2 + dy**2 + dz**2)**0.5

          hij = 0.5 * (h_pt[i] + h[j])

          sig = 1.0/np.pi
          q = rr / hij

          WIij = 0.0

          if q <= 1.0:
            WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

          if (q > 1.0) and (q <= 2.0):
            WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
          
          s += m[j] * ionFracx[j, k] / rho[j] * WIij
          ss += m[j] / rho[j] * WIij # For normalization!

      if ss == 0:
        iFc[i, k] = 1e-15
      else:
        iFc[i, k] = s / ss

  return iFc


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


# 0    1   2    3    4    5    6      7      8   9    10   11    12   13
# HI  HII  CI  CII CIII  CIV  SiII  SiIII  SiIV  NV  OVI  FeII  MgI  MgII

with open('Clumps.pkl', 'rb') as f:
  dictx = pickle.load(f)
  clumps = dictx['nxclumps']

print()
print(f'There are {len(clumps)} clumps in the file!')
print()


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

print('sort(mass) = ', np.sort(mass))

x = x[n]
y = y[n]
z = z[n]

r = np.vstack((x, y, z))
r = np.transpose(r)
print(r.shape)

vx = vx[n]
vy = vy[n]
vz = vz[n]

v = np.vstack((vx, vy, vz))
v = np.transpose(v)
print(v.shape)

rho_cgs = rho * unit_rho
nH_cgs = rho_cgs * XH / mH


clump = clumps[0] # ----> the first clump !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(f'min(h[clump]) = {min(h[clump]):.6f}, max(h[clump]) = {max(h[clump]):.6f}, median(h[clump]) = {np.median(h[clump]):.6f}, , mean(h[clump]) = {np.mean(h[clump]):.6f}')


nH_cgs_clump = nH_cgs[clump]

nt = clump[np.where(nH_cgs_clump == max(nH_cgs_clump))[0][0]] # To make sure that the line of sight passes through the densest region of the clump!

print('max nH coords = ', (x[nt], y[nt], z[nt]))
print('max nH = ', nH_cgs[nt])
print()

#for ndx in clump:
#  print(ionFrac[ndx, 0], ionFrac[ndx, 1], ionFrac[ndx, 0]+ionFrac[ndx, 1])
#s()

#rtmp = np.sqrt((x - x[nt])**2 + (y - y[nt])**2 + (z - z[nt])**2)
#ntmp = np.argsort(rtmp)
#ntmp = ntmp[:2000]

ntmp = clump # contains the index of the particles in the clump.

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

h_pt = np.zeros(len(xt)) + np.mean(h[clump])
print('h_pt = ', h_pt[0])


rrt = np.sqrt(pt[:, 0] * pt[:, 0] + pt[:, 1] * pt[:, 1] + pt[:, 2] * pt[:, 2])
#-------------------------------------------------

#----
plt.scatter(x[ntmp], y[ntmp], s = 1, color = 'k')
plt.scatter(xt, yt, color = 'red', s = 10) #----- Line of Sight ---------
plt.plot(xt, yt, color = 'lime') #----- Line of Sight ---------
delta = 0.02
plt.xlim(x[nt] - delta, x[nt]+delta)
plt.ylim(y[nt] - delta, y[nt]+delta)
plt.show()
#----

'''
iFrac = getIonFrac(Typ, r, pt, rho, mass, h, h_pt, ionFrac[:, :])
print()
print('iFrac generated !')
print()


#for i in range(2000):
#  print(iFrac[i, 1])

#for ndx in range(len(tt)):
#  print(iFrac[ndx, 0], iFrac[ndx, 1], iFrac[ndx, 0]+iFrac[ndx, 1])

plt.scatter(tt, iFrac[:, 0], s = 5)
plt.show()
'''

rho_cgs = getDensity(Typ, r, pt, mass, h, h_pt)
nH_pt = rho_cgs * unit_rho * XH / mH

#                     H     C      Mg     Si     Fe     N      O
#SolRatio = np.array([0.0, -3.57, -4.40, -4.49, -4.50, -4.17, -3.31])

ElmID = ['HI', 'HII', 'CI', 'CII', 'CIII', 'CIV', 'SiII', 'SiIII', 'SiIV', 'NV', 'OVI', 'FeII', 'MgI', 'MgII']

#            HI  HII    CI    CII    CIII   CIV    SiII  SiIII   SiIV    NV     OVI   FeII   MgI    MgII
SolRatio = [0.0, 0.0, -3.57, -3.57, -3.57, -3.57, -4.49, -4.49, -4.49, -4.17, -3.31, -4.50, -4.40, -4.40]
SolRatio = np.array(SolRatio)

SolRatio = 10**SolRatio

Ncol = np.zeros(len(SolRatio))

for k in range(len(SolRatio)):
  for i in range(len(rrt)-1):
    dl = (rrt[i+1] - rrt[i]) * unit_length
    Ncol[k] += nH_pt[i] * dl * iFrac[i, k]

print()
for i in range(len(SolRatio)):
  print(f'log(N({ElmID[i]})) = {np.log10(Ncol[i])}', )
print()




