
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('./Outputs/*.bin'))

Nfiles = len(filez)

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the entire file at once

    # Use a memoryview to avoid copies
    buffer = memoryview(file_content)

    # Unpack N and N_ionFrac
    N, N_ionFrac = struct.unpack_from('ii', buffer)
    offset = 8  # Start after the first two integers

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

    # Read arrays
    Typ = read_array(np.int32, N, 4)
    x = read_array(np.float32, N, 4)
    y = read_array(np.float32, N, 4)
    z = read_array(np.float32, N, 4)
    vx = read_array(np.float32, N, 4)
    vy = read_array(np.float32, N, 4)
    vz = read_array(np.float32, N, 4)
    rho = read_array(np.float32, N, 4)
    h = read_array(np.float32, N, 4)
    u = read_array(np.float32, N, 4)
    mass = read_array(np.float32, N, 4)
    ionFrac = read_array(np.float32, N_ionFrac, 4)
    ngbDebug = read_array(np.int32, N, 4)

    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug


jj = 981628 #923810


nHArr = np.zeros(Nfiles)
TArr = np.zeros(Nfiles)

res = []

plt.figure(figsize = (18, 8))

check = 0

N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filez[0]) #!!! Just to get REAL N
N = len(np.where(u != 0.0)[0])

for i in range(0, len(filez), 1):
  Nn, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filez[i])

  n = np.where(u != 0.0)[0]
  
  rho = rho[n]
  u = u[n]

  h = h[n]
  x = x[n]
  y = y[n]
  z = z[n]
  
  nz = np.where(np.abs(z) < 0.01)[0]
  
  xx = x[nz]
  yy = y[nz]
  zz = z[nz]
  
  #nt = np.where((x > 0.03) & (x < 0.035) & (np.abs(y) < 0.01) & (np.abs(z) < 0.05))[0]
  #print(nt)
  #s()

  vx = vx[n]
  vy = vy[n]
  vz = vz[n]

  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u
  
  XH = 0.7
  nH = rho * unit_rho * XH /mH
  
  #print(np.where(np.log10(nH) > 2.5))
  
  print(np.where((np.log10(Temp) < 3.8) & (np.abs(z) < 0.03)))
  
  try:
    nHArr[i] = np.log10(nH[jj])
    TArr[i] = np.log10(Temp[jj])
  except:
    pass  

  plt.clf()

  # First subplot for Temp vs nH
  plt.subplot(1, 2, 1)  # (1 row, 2 columns, first subplot)
  plt.scatter(np.log10(nH), np.log10(Temp), s=0.01, color='k')
  plt.scatter(nHArr, TArr, s=50, color='r')
  plt.xlim(-1.3, 3)
  plt.ylim(3, 11)
  plt.xlabel('nH')
  plt.ylabel('Temperature')
  plt.title(f'Temp vs nH for File: {filez[i]}')

  # Second subplot for y vs x
  
  NN = len(np.where(u != 0.0)[0])
  print("N, NN = ", N, NN)
  
  xxx = x[N:NN]
  yyy = y[N:NN]
  zzz = z[N:NN]
  nnnz = np.where(np.abs(zzz) < 0.01)[0]
  xxx = xxx[nnnz]
  yyy = yyy[nnnz]
  zzz = zzz[nnnz]
  
  
  plt.subplot(1, 2, 2)  # (1 row, 2 columns, second subplot)
  plt.scatter(xx, yy, s=0.5, color='k')
  plt.scatter(xxx, yyy, s=1.0, color='b')
  try:
    plt.scatter(x[jj], y[jj], s=20, color='r')
    print(x[jj], y[jj], z[jj])
  except:
    pass

  xy = 0.015
  plt.xlim(-xy, xy)
  plt.ylim(-xy, xy)

  plt.pause(0.01)  # Pause to render the plots

  plt.draw()

plt.savefig('out.png')

plt.show() # showing the last plot at the end!







