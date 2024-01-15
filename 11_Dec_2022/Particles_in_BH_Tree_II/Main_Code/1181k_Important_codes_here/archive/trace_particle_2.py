
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('./Outputs/*.bin'))


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


jj = 923810

res = []

plt.figure(figsize = (9, 8))

for i, filename in enumerate(filez):
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]

  h = h[n]
  x = x[n]
  y = y[n]
  z = z[n]
  
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
  
  plt.clf()

  # Create scatter plot
  plt.scatter(np.log10(nH), np.log10(Temp), s = 0.01, color = 'k')
  plt.scatter(np.log10(nH[jj]), np.log10(Temp[jj]), s = 50, color = 'r')
  
  plt.xlim(-1.3, 3)
  plt.ylim(3, 11)
  
  plt.xlabel('nH')
  plt.ylabel('Temperature')
  plt.title(f'Scatter Plot for File: {filename}')
  plt.pause(0.2)  # Pause for a short period to render the plot

  plt.draw()

plt.savefig('out.png')

plt.show() # showing the last plot at the end!







