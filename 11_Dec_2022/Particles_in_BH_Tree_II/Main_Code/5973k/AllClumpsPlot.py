
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob


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


filez = np.sort(glob.glob('./Outputs/*.bin'))


with open('Clumps.pkl', 'rb') as f:
  dictx = pickle.load(f)
  clumps = dictx['nxclumps']

print()
print(f'There are {len(clumps)} clumps in the file!')
print()

j = 2
print(len(clumps[j]))
s()

nx = []

for clump in clumps:
  nx += clump

plt.figure(figsize = (19, 5))


for i in range(0, len(filez), 4):
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filez[i])

  n = np.where(u != 0.0)[0]

  x = x[n]
  y = y[n]
  z = z[n]
  
  rho = rho[n]

  plt.clf()
  
  delta = 0.04

  plt.subplot(1, 3, 1)
  #plt.scatter(x, y, s = 0.01, color = 'k')
  plt.scatter(x[nx], y[nx], s = 0.05, color = 'orange')
  medx = np.median(x[nx])
  medy = np.median(y[nx])
  #plt.xlim(medx-delta, medx+delta)
  #plt.ylim(medy-delta, medy+delta)
  
  xy = 0.72
  
  plt.xlim(-xy, xy)
  plt.ylim(-xy, xy)



  plt.subplot(1, 3, 2)
  #plt.scatter(x, z, s = 0.01, color = 'k')
  plt.scatter(x[nx], z[nx], s = 0.05, color = 'orange')
  medx = np.median(x[nx])
  medz = np.median(z[nx])
  #plt.xlim(medx-delta, medx+delta)
  #plt.ylim(medz-delta, medz+delta)
  
  plt.xlim(-xy, xy)
  plt.ylim(-xy, xy)
  
  
  plt.subplot(1, 3, 3)
  #plt.scatter(x, z, s = 0.01, color = 'k')
  plt.scatter(y[nx], z[nx], s = 0.05, color = 'orange')
  medx = np.median(y[nx])
  medz = np.median(z[nx])
  #plt.xlim(medx-delta, medx+delta)
  #plt.ylim(medz-delta, medz+delta)
  
  plt.xlim(-xy, xy)
  plt.ylim(-xy, xy)
  
  
  plt.pause(0.1)  # Pause to render the plots
  plt.draw()

plt.savefig('AllClumps.png')



