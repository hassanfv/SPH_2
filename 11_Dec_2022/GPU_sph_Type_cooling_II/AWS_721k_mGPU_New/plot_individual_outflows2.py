
import numpy as np
import matplotlib.pyplot as plt
import glob


filez = glob.glob('./Out712k_mgpu/*.bin')


import struct

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



plt.figure(figsize=(10, 8))

j = 0

for filename in filez:

  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)
  
  print('j = ', j)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]
  h = h[n]


  x = x[n]
  y = y[n]
  z = z[n]

  vx = vx[n]
  vy = vy[n]
  vz = vz[n]


  ndx_BH = 701166
  xx = x[ndx_BH:]
  yy = y[ndx_BH:]
  zz = z[ndx_BH:]
  uu = u[ndx_BH:]

  vxx = vx[ndx_BH:]
  vyy = vy[ndx_BH:]
  vzz = vz[ndx_BH:]


  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24

  unit_u = 7.02136e+12
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * uu * unit_u

  #---- Last outflow particles:
  xL = xx[-2:]
  yL = yy[-2:]
  zL = zz[-2:]

  vxL = vxx[-2:]
  vyL = vyy[-2:]
  vzL = vzz[-2:]



  #scatter = plt.scatter(xx, yy, color = 'blue', s=30)
  scatter = plt.scatter(xL, yL, color = 'orange', s=20)

  j+= 1
  
  #if j > 20:
  #  break





#----- drawing a circle ----
R = 0.1635
center = (0, 0)
theta = np.linspace(0, 2*np.pi, 100)
xc = center[0] + R * np.cos(theta)
yc = center[1] + R * np.sin(theta)

plt.plot(xc, yc, color = 'blue', linewidth = 5, alpha = 0.4)

plt.axvline(x = 0, linestyle = '--', color = 'b')
plt.axhline(y = 0, linestyle = '--', color = 'b')
#---------------------------

xy = 0.01

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y, colored by T value')

plt.savefig('fig1.png')

plt.show()



