
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import imageio
import time


filez = np.sort(glob.glob('./Outputs/*.bin'))

Nfiles = len(filez)

unit_velocity_cgs = 2.31909e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 5.37816e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 8.46128e-24 # !!!!!!!!!!!!!!!!!!!

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        file_content = file.read()
    
    buffer = memoryview(file_content)
    offset = 0

    # Function to read and advance the offset
    def read_array(dtype, size, itemsize):
        nonlocal offset
        array = np.frombuffer(buffer, dtype=dtype, count=size, offset=offset)
        offset += size * itemsize
        return array

    # Read N
    N = np.frombuffer(buffer, dtype=np.int32, count=1, offset=offset)[0]
    offset += 4  # Size of int32

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

    return x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N


TA = time.time()

jj = 2564636  # [2177937 2564636 2759865 2763893] [2161729 2356983 2548431 2739641]


output_folder = './Plots_for_Animation/'
image_files = []

nHArr = np.zeros(Nfiles)
TArr = np.zeros(Nfiles)

res = []

plt.figure(figsize = (12, 12))

check = 0

x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N = read_binary_file(filez[0]) #!!! Just to get REAL N
N = len(np.where(u != 0.0)[0])

ii = 0

for i in range(0, len(filez), 5):
  x, y, z, vx, vy, vz, rho, h, u, mass, Typ, Nn = read_binary_file(filez[i])

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
  
  #nt = np.where((x > 0.04) & (x < 0.045) & (np.abs(y) < 0.005) & (np.abs(z) < 0.005))[0]
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
  
  #print(np.where((np.log10(Temp) < 5.2) & (np.log10(nH) >= 1.7) & (x > 0.01) & (x < 0.04) & (np.abs(y) < 0.04) & (np.abs(z) < 0.04)))
  
  try:
    nHArr[i] = np.log10(nH[jj])
    TArr[i] = np.log10(Temp[jj])
  except:
    pass  

  NN = len(np.where(u != 0.0)[0])
  #print("N, NN = ", N, NN)

  plt.clf()

  # First subplot for Temp vs nH
  plt.subplot(2, 2, 1)  # (2 row, 2 columns, first subplot)
  plt.scatter(np.log10(nH), np.log10(Temp), s=0.01, color='k')
  try:
    plt.scatter(np.log10(nH[N:NN]), np.log10(Temp[N:NN]), s=1.0, color='b', alpha = 0.4)
  except:
    pass
  plt.scatter(nHArr, TArr, s=30, color='r')
  plt.xlim(-2.0, 4.5)
  plt.ylim(1.5, 11)
  plt.xlabel('nH')
  plt.ylabel('Temperature')
  plt.title(f'Temp vs nH for File: {filez[i]}')

  # Second subplot for y vs x
  xxx = x[N:NN]
  yyy = y[N:NN]
  zzz = z[N:NN]
  nnnz = np.where(np.abs(zzz) < 0.01)[0]
  xxx = xxx[nnnz]
  yyy = yyy[nnnz]
  zzz = zzz[nnnz]
  
  
  plt.subplot(2, 2, 2)  # (2 row, 2 columns, second subplot)
  plt.scatter(xx, yy, s=0.5, color='k')
  plt.scatter(xxx, yyy, s=1.0, color='b')
  try:
    plt.scatter(x[jj], y[jj], s=30, color='r')
    #print(x[jj], y[jj], z[jj])
  except:
    pass
  xy = 0.30
  plt.xlim(-xy, xy)
  plt.ylim(-xy, xy)



  rr = (x*x + y*y + z*z)**0.5
  vr = (vx*vx + vy*vy + vz*vz)**0.5 * unit_velocity_cgs / 100/1000
  
  plt.subplot(2, 2, 3) # (2 row, 2 columns, third subplot)
  plt.scatter(rr, vr, s = 2, color = 'k')
  try:
    plt.scatter(rr[N:NN], vr[N:NN], s = 5, color = 'b')
  except:
    pass
  
  try:
    plt.scatter(rr[jj], vr[jj], s = 30, color = 'r')
  except:
    pass
  plt.xlim(0, 0.30)
  plt.ylim(-2000, 32000)
  #plt.ylim(29750, 30250)
  #plt.yscale('log')
  
  plt.axhline(y = 30000, color = 'b')
  
  filename = f'{output_folder}plot_{ii}.png'  # Filename for the plot
  plt.savefig(filename)
  image_files.append(filename)  # Add filename to the list
  ii += 1

  plt.pause(0.01)  # Pause to render the plots
  plt.draw()

plt.savefig('out.png')

#plt.show() # showing the last plot at the end!

print('Elapsed time = ', time.time() - TA)





