import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pickle


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

  iFc = np.zeros((M, 14)) # iFc = ionFraction at the pt positions! pt contains the positions along the line of sight!

  for i in range(M):

    for k in range(14): # k represents the 14 species we have in our ionFrac.

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
          
        s += m[j] * ionFracx[j, k] / rho[j] * WIij

      iFc[i, k] = s

  return iFc


#===== getPE
@njit
def getPE(r, m, G):
  
  N = len(r)
  PE = np.zeros(N)
  
  for i in range(N):
    for j in range(N):
      if i != j:
        rr = np.linalg.norm(r[i] - r[j])
        PE[i] -= G * m[i] * m[j] / rr

  return PE


#===== getCMvel (center-of-mass velocity)
def getCMvel(m, v):

  N = len(m)
  totMass = np.sum(m)
  
  sx = 0.0
  sy = 0.0
  sz = 0.0
  for i in range(N):
    sx += m[i] * v[i, 0]
    sy += m[i] * v[i, 1]
    sz += m[i] * v[i, 2]

  vxcm = sx / totMass
  vycm = sy / totMass
  vzcm = sz / totMass
  
  return [vxcm, vycm, vzcm]



#===== getKE
def getKE(m, v):
  
  N = len(m)
  KE = np.zeros(N)
  
  vxcm, vycm, vzcm = getCMvel(m, v)
  
  for i in range(N):
    vx = v[i, 0] - vxcm
    vy = v[i, 1] - vycm
    vz = v[i, 2] - vzcm
    KE[i] = 0.5 * m[i] * (vx*vx + vy*vy + vz*vz) # Kinetic Energy = 1/2 m*v^2
  
  return KE



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

v = np.vstack((vx, vy, vz))
v = np.transpose(v)
print(v.shape)

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
ntmp = ntmp[:500]

#--- Output to a file
dictx = {'r': r, 'nx': ntmp}
with open('group_index.pkl', 'wb') as f:
  pickle.dump(dictx, f)

s()

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

G = 1.0
PE = getPE(r[ntmp, :], mass[ntmp], G)
print(f'tot PE = {np.sum(PE):3E}')

KE = getKE(mass[ntmp], v[ntmp, :])
print(f'tot KE = {np.sum(KE):3E}')
print()
print(f'tot KE + PE = {(np.sum(KE)+np.sum(PE)):3E}')
print()

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
plt.scatter(x[ntmp], y[ntmp], s = 1, color = 'k')
plt.plot(xt, yt, color = 'red') #----- Line of Sight ---------
delta = 0.02
plt.xlim(x[nt] - delta, x[nt]+delta)
plt.ylim(y[nt] - delta, y[nt]+delta)
plt.show()
#----

#---- plot --> nH_cgs vs T
plt.scatter(nH_cgs[ntmp], np.log10(Temp[ntmp]), s = 0.5)
plt.show()



n_ion = 0 # HI
iFrac = getIonFrac(r, pt, rho, mass, h, ionFrac[:, :])
T_pt = getTemp(r, pt, rho, Temp, mass, h)

#----- plot ------
#plt.scatter(rrt, nH_pt, s = 5)
plt.scatter(rrt, np.log10(T_pt), s = 5, label = 'T vs. r')
#plt.ylim(3, 11)
plt.legend()
plt.show()


plt.scatter(iFrac[:, 0], T_pt, s = 5, color = 'k', label = 'HI-Frac vs. T')
plt.xscale('log')
plt.yscale('log')
#plt.ylim(3, 11)
plt.legend()
plt.show()



#----- plot ------
plt.scatter(rrt, iFrac[:, 0], s = 5, color = 'k', label = 'HI-Frac vs. r')
plt.yscale('log')
plt.legend()
plt.show()


rho_cgs = getDensity(r, pt, mass, h) * unit_rho
nH_pt = rho_cgs * XH / mH

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






