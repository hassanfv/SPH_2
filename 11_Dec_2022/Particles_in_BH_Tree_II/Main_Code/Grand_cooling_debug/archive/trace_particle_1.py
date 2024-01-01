
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



jj = 875985   # [874002 875985 877706 877710 939445 940947]

res = []

for i, filename in enumerate(filez):
  N, Typ, x, y, z, vx, vy, vz, rho, h, u, mass = readBinaryFile(filename)

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
  
  vr = np.sqrt(vx*vx + vy*vy + vz*vz)

  kB = 1.3807e-16
  mu = 0.61
  mH = 1.673534e-24
  gamma = 5./3.
  Temp = (gamma - 1) * mH / kB * mu * u * unit_u

  res.append([i, np.log10(Temp[jj]), rho[jj], h[jj], vr[jj]*unit_velocity_cgs/100/1000])
  
  ntmp = np.where((Temp > 6000000) & (Temp < 6500000) & (x > 0.0) & (y > 0.0) & (np.abs(z) < 0.035))[0]
  print('ntmp = ', ntmp)
  print()


res = np.array(res)

t = res[:, 0]
T = res[:, 1]
rho = res[:, 2]
h = res[:, 3]
vr= res[:, 4]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust figsize as needed

# Scatter plot for each subplot
axs[0, 0].scatter(t, T, s=20)
axs[0, 0].set_title("Temperature")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("Log10(Temp)")

axs[0, 1].scatter(t, rho, s=20)
axs[0, 1].set_title("Density")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("rho")

axs[1, 0].scatter(t, h, s=20)
axs[1, 0].set_title("Scale Height")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("h")

axs[1, 1].scatter(t, vr, s=20)
axs[1, 1].set_title("Radial Velocity")
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("vr")

# Adjust layout
plt.tight_layout()

plt.savefig('figure.png')

# Show plot
plt.show()







