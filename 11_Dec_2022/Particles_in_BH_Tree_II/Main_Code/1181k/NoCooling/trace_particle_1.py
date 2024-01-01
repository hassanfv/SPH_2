
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('*.bin'))


unit_velocity_cgs = 3.21138e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.0313e+13 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 1.62251e-23 # !!!!!!!!!!!!!!!!!!!

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


# [ 961169  973175  997208  997829 1009723 1009847 1022249 1022250 1262768 1265876 1384175 1390297 1399028 1408127 1645235 1645296 1778357]

jj = 1778596 #742726

res = []

for i, filename in enumerate(filez):
  N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, ngbDebug = readBinaryFile(filename)

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

  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u

  res.append([i, np.log10(Temp[jj]), rho[jj], h[jj]])
  
  ntmp = np.where((Temp > 6000000) & (Temp < 6500000) & (x < 0.0) & (y < 0.0) & (np.abs(z) < 0.035))[0]
  print('ntmp = ', ntmp)


res = np.array(res)

x = res[:, 0]
T = res[:, 1]
rho = res[:, 2]
h = res[:, 3]

#plt.scatter(x, T, s = 20)
#plt.scatter(x, rho, s = 20)
plt.scatter(x, h, s = 20)

plt.show()







