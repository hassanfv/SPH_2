import numpy as np
import matplotlib.pyplot as plt
import glob
import struct

#filz = np.sort(glob.glob('./No_cooling_Outputs/*.bin'))

filz = np.sort(glob.glob('./Out470k/*.bin'))


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


res = []

for j, filename in enumerate(filz):

  # Usage
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]

  x = x[n]
  y = y[n]
  z = z[n]

  nz = np.where(np.abs(z) < 111110.06)[0]

  x = x[nz]
  y = y[nz]
  z = z[nz]
  
  u = u[nz]
  rho = rho[nz]
  
  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24


  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u
  #TempBAd = (gamma - 1) * mH / kB * mu * uBAd * unit_u
  #TempAC = (gamma - 1) * mH / kB * mu * uAC * unit_u
  
  #print(np.sort(Temp))
  #print((np.where(Temp < 1.0)))
  
  nnt = 58850
  
  XH = 0.7
  nH = rho * unit_rho * XH / mH
  
  try:
    res.append([j, rho[nnt], u[nnt], x[nnt], y[nnt], z[nnt], nH[nnt]])
  except:
    pass


res = np.array(res)

t = res[:, 0]
rho = res[:, 1]
u = res[:, 3]

xx = res[:, 5]
yy = res[:, 6]
zz = res[:, 7]

nH = res[:, 8]


#---------------------
u_t = 11.0
Temp = (gamma - 1) * mH / kB * mu * u_t * unit_u
print(Temp)
print(np.log10(Temp))



Temp = (gamma - 1) * mH / kB * mu * u * unit_u

plt.scatter(t, nH, s = 5, color = 'k')
#plt.scatter(t, np.log10(Temp), s = 5, color = 'k')

plt.savefig('rho_or_u_evolution.png')

plt.show()

scatter = plt.scatter(x, z, s=0.01)
plt.scatter(xx, zz, s = 2, color = 'lime')

xyran = 0.26

plt.xlim(-xyran, xyran)
plt.ylim(-xyran, xyran)

plt.colorbar(scatter, label='T Value')

plt.show()







