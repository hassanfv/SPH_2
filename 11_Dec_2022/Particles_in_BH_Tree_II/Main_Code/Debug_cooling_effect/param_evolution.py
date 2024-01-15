
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import imageio
import time
import pandas as pd

filez = np.sort(glob.glob('./Outputs_1181X/*.bin'))

#filez = np.sort(glob.glob('./Outputs/*.bin'))

Nfiles = len(filez)

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

def readBinaryFile(filename):
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
    
    dudt = read_array(np.float32, N, 4)
    dudt_pre = read_array(np.float32, N, 4)
    uBad = read_array(np.float32, N, 4)
    uAad = read_array(np.float32, N, 4)
    u = read_array(np.float32, N, 4)
    
    mass = read_array(np.float32, N, 4)

    return x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, N


TA = time.time()

jj = 763275

output_folder = './' + str(jj) + '_Plots_for_Animation/'
image_files = []

nHArr = np.zeros(Nfiles)
TArr = np.zeros(Nfiles)

grid = np.zeros(Nfiles)
vArr = np.zeros(Nfiles)

res = []

plt.figure(figsize = (12, 12))

check = 0

x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, N = readBinaryFile(filez[0]) #!!! Just to get REAL N
N = len(np.where(u != 0.0)[0])

res = []

for i in range(0, len(filez), 1):
  x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, Nn = readBinaryFile(filez[i])

  #print(f'{i+1} out of {Nfiles}')

  n = np.where(u != 0.0)[0]
  
  rho = rho[n]
  
  dudt = dudt[n]
  dudt_pre = dudt_pre[n]
  uBad = uBad[n]
  uAad = uAad[n]
  u = u[n]
  
  dt = 4e-8
  
  ut = dudt[jj] * dt
  ut_pre = dudt_pre[jj] * dt
  uB = uBad[jj]
  uA = uAad[jj]
  uC = u[jj] # this is actually uAC
  ut_c = uC - uA

  h = h[n]
  x = x[n]
  y = y[n]
  z = z[n]
  
  dist = (x*x + y*y + z*z)**0.5
  
  vx = vx[n]
  vy = vy[n]
  vz = vz[n]
  
  vv = (vx*vx + vy*vy + vz*vz)**0.5 * unit_velocity_cgs / 100/1000

  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u
  
  XH = 0.7
  nH = rho * unit_rho * XH /mH

  rjj = dist[jj]
  nHjj = (nH[jj])
  vjj = vv[jj]
  Tjj = np.log10(Temp[jj])
  Pjj = (gamma - 1.0) * rho[jj] * u[jj];
  
  print(f'rkpc = {rjj:.3f}, v = {vjj:.3f}, nH = {nHjj:.2f}, ut = {ut:.4f}, ut_pre = {ut_pre:.4f}, ut_c = {ut_c:.4f}, uBad = {uB:.4f}, uAad = {uA:.4f}, uAC = {uC:.4f}')
  
  res.append([i, rjj, nHjj, vjj, Tjj, Pjj, ut, ut_pre, ut_c, uB, uA, uC])


res = np.array(res)
df = pd.DataFrame(res)
df.columns = ['t', 'r', 'nH', 'v', 'T', 'P', 'ut', 'ut_pre', 'ut_c', 'uBad', 'uAad', 'uAC']

print()
print(df)

df.to_csv('X-' + str(jj) + '.csv', index = False)

print('Elapsed time = ', time.time() - TA)





