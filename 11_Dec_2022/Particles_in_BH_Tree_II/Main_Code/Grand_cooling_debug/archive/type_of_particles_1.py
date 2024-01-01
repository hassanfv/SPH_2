
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('./Outputs/*.bin'))


unit_velocity_cgs = 3.21138e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.0313e+13 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 1.62251e-23 # !!!!!!!!!!!!!!!!!!!



def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the entire file at once

    buffer = memoryview(file_content)

    # Unpack N
    N = struct.unpack_from('i', buffer, 0)[0]
    offset = 4  # Start after the first integer

    # Function to create NumPy array from buffer
    def create_array(dtype, count, offset):
        dtype_size = np.dtype(dtype).itemsize
        return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset), offset + dtype_size * count

    # Read arrays
    Typ, offset = create_array(np.int32, N, offset)
    x, offset = create_array(np.float32, N, offset)
    y, offset = create_array(np.float32, N, offset)
    z, offset = create_array(np.float32, N, offset)
    vx, offset = create_array(np.float32, N, offset)
    vy, offset = create_array(np.float32, N, offset)
    vz, offset = create_array(np.float32, N, offset)
    rho, offset = create_array(np.float32, N, offset)
    h, offset = create_array(np.float32, N, offset)
    u, offset = create_array(np.float32, N, offset)
    mass, offset = create_array(np.float32, N, offset)

    return N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass


kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24


jj = 875985   # [874002 875985 877706 877710 939445 940947]

res = []

for i, filename in enumerate(filez):
  N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass = readBinaryFile(filename)

  nxBH = np.where(mass == max(mass))[0][0]
  
  #print('nxBH = ', nxBH)

  n = np.where(u != 0.0)[0]
  print('non-zero u = ', len(n))
  
  rho = rho[n]
  rho_cgs = rho * unit_rho
  XH = 0.7
  nH = rho_cgs * XH / mH
  
  u = u[n]

  h = h[n]
  
  x = x[n]
  y = y[n]
  z = z[n]

  vx = vx[n]
  vy = vy[n]
  vz = vz[n]
  
  xj = x[jj]
  yj = y[jj]
  zj = z[jj]
  
  dx = x - xj
  dy = y - yj
  dz = z - zj
  
  dist = np.sqrt(dx*dx + dy*dy + dz*dz)
  
  nh = np.where(dist <= 2.0 * h[jj])[0]
  print('(dist[nh]) = ', (dist[nh]))
  
  print('(nH[nh]) = ', (nH[nh]))
  
  #--- determining which particle (i.e. gas or outflow) are withing the smoothing length
  NGas = len(np.where(nh < nxBH)[0])
  NOutflow = len(np.where(nh > nxBH)[0])

  print(f'N_Gas = {NGas},  N_Outflow = {NOutflow},  Nngb = {NGas + NOutflow},  h = {h[jj]},  nH = {nH[jj]}')
  print()



