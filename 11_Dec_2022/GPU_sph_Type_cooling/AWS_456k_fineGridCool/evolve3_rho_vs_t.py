import numpy as np
import matplotlib.pyplot as plt
import glob

#filz = np.sort(glob.glob('./No_cooling_Outputs/*.bin'))

filz = np.sort(glob.glob('./Out456k/*.bin'))


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


res = []

for j, filename in enumerate(filz):

  # Usage
  N, Typ, x, y, z, vx, vy, vz, rho, h, u, uBAd, uAC, mass, dudt, utprevious = loadArraysFromBinary(filename)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]
  uBAd = uBAd[n]
  uAC = uAC[n]
  
  dudt = dudt[n]
  utprevious = utprevious[n]

  x = x[n]
  y = y[n]
  z = z[n]
  
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
  uBAd = uBAd[nz]
  uAC = uAC[nz]
  rho = rho[nz]
  
  dudt = dudt[nz]
  utprevious = utprevious[nz]


  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24

  unit_u = 4100904397311.213
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u
  TempBAd = (gamma - 1) * mH / kB * mu * uBAd * unit_u
  TempAC = (gamma - 1) * mH / kB * mu * uAC * unit_u
  
  #print(np.sort(Temp))
  #print((np.where(Temp < 1.0)))
  
  nnt = 145251
  
  XH = 0.7
  unit_rho = 6.451817553342665e-24
  nH = rho * unit_rho * XH / mH
  
  try:
    res.append([j, rho[nnt], uBAd[nnt], u[nnt], uAC[nnt], x[nnt], y[nnt], z[nnt], nH[nnt]])
    print(j, rho[nnt], uBAd[nnt], u[nnt], uAC[nnt], x[nnt], y[nnt], z[nnt], nH[nnt])
    #res.append([j, Temp[nnt], TempBAd[nnt], TempAC[nnt], x[nnt], y[nnt], z[nnt], dudt[nnt], u[nnt], uBAd[nnt], uAC[nnt], utprevious[nnt]])
    #print('j = ', j)
  except:
    pass


res = np.array(res)

t = res[:, 0]
rho = res[:, 1]
uBAd = res[:, 2]
u = res[:, 3]
uAC = res[:, 4]

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







