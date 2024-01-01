import numpy as np
import matplotlib.pyplot as plt
import glob
import time

#filz = np.sort(glob.glob('./No_cooling_Outputs/*.bin'))

filz = np.sort(glob.glob('./Out456k/*.bin'))

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



res = []

for j, filename in enumerate(filz):

  # Usage
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]
  
  ionFrac = ionFrac.reshape((N, 14))

  x = x[n]
  y = y[n]
  z = z[n]
  
  ionFrac = ionFrac[n]
  
  #vx = vx[n]
  #vy = vy[n]
  #vz = vz[n]

  nz = np.where(np.abs(z) < 111110.06)[0]

  x = x[nz]
  y = y[nz]
  z = z[nz]
  
  #vx = vx[nz]
  #vy = vy[nz]
  #vz = vz[nz]

  u = u[nz]
  rho = rho[nz]
  
  ionFrac = ionFrac[nz]


  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24

  unit_u = 4100904397311.213
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u
  
  nnt = 229743
  
  XH = 0.7
  unit_rho = 6.451817553342665e-24
  nH = rho * unit_rho * XH / mH
  
  try:
    print(j)
    res.append([j, rho[nnt], u[nnt], x[nnt], y[nnt], z[nnt], nH[nnt], ionFrac[nnt, 0]])
  except:
    pass


res = np.array(res)

t = res[:, 0]
rho = res[:, 1]
u = res[:, 2]

xx = res[:, 3]
yy = res[:, 4]
zz = res[:, 5]

nH = res[:, 6]

HI_frac = res[:, 7]
#---------------------

plt.scatter(t, HI_frac, s = 5, color = 'k')

plt.yscale('log')

plt.show()

s()

Temp = (gamma - 1) * mH / kB * mu * u * unit_u

#plt.scatter(t, nH, s = 5, color = 'k')
plt.scatter(t, np.log10(Temp), s = 5, color = 'k')

plt.savefig('rho_or_u_evolution.png')

plt.show()

scatter = plt.scatter(x, z, s=0.01)
plt.scatter(xx, zz, s = 2, color = 'lime')

xyran = 0.36

plt.xlim(-xyran, xyran)
plt.ylim(-xyran, xyran)

plt.colorbar(scatter, label='T Value')

plt.show()







