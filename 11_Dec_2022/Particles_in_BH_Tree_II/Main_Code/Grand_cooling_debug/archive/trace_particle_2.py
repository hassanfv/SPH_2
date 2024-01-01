
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
        file_content = f.read()

    # Create a memoryview from the file content
    buffer = memoryview(file_content)

    # Function to read a chunk of data and update the offset
    def read_chunk(dtype, count, offset):
        dtype_size = np.dtype(dtype).itemsize
        data = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        offset += dtype_size * count
        return data, offset

    # Initialize offset
    offset = 0

    # Read N
    N, offset = read_chunk(np.int32, 1, offset)
    N = N[0]  # Extract the single integer value

    # Read arrays
    Typ, offset = read_chunk(np.int32, N, offset)
    x, offset = read_chunk(np.float32, N, offset)
    y, offset = read_chunk(np.float32, N, offset)
    z, offset = read_chunk(np.float32, N, offset)
    vx, offset = read_chunk(np.float32, N, offset)
    vy, offset = read_chunk(np.float32, N, offset)
    vz, offset = read_chunk(np.float32, N, offset)
    rho, offset = read_chunk(np.float32, N, offset)
    h, offset = read_chunk(np.float32, N, offset)
    uB, offset = read_chunk(np.float32, N, offset)
    uA, offset = read_chunk(np.float32, N, offset)
    u, offset = read_chunk(np.float32, N, offset)
    mass, offset = read_chunk(np.float32, N, offset)

    return x, y, z, vx, vy, vz, rho, h, uB, uA, u, mass, Typ, N



jj = 875985   # [874002 875985 877706 877710 939445 940947]

res = []

for i, filename in enumerate(filez):
  x, y, z, vx, vy, vz, rho, h, uB, uA, u, mass, Typ, N = readBinaryFile(filename)

  n = np.where(u != 0.0)[0]
  rho = rho[n]
  u = u[n]

  h = h[n]
  
  x = x[n]
  y = y[n]
  z = z[n]
  
  nn = np.where(np.abs(z) < 0.035)[0]

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

T_norm = (T - np.min(T)) / (np.max(T) - np.min(T))
rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))
h_norm = (h - np.min(h)) / (np.max(h) - np.min(h))
vr_norm = (vr - np.min(vr)) / (np.max(vr) - np.min(vr))

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figsize as needed

# Scatter plot for each subplot
axs[0, 0].scatter(t, T, s=20)
axs[0, 0].set_title("Temperature")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("Log10(Temp)")

axs[0, 1].scatter(t, rho, s=20)
axs[0, 1].set_title("Density")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("rho")

axs[0, 2].scatter(t, h, s=20)
axs[0, 2].set_title("Scale Height")
axs[0, 2].set_xlabel("t")
axs[0, 2].set_ylabel("h")

axs[1, 0].scatter(t, vr, s=20)
axs[1, 0].set_title("Radial Velocity")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("vr")

# New scatter plot
axs[1, 1].scatter(x[nn], y[nn], s=0.01, color='k')
axs[1, 1].scatter(x[jj], y[jj], s=20, color='r')
axs[1, 1].set_xlim(0, 0.30)
axs[1, 1].set_ylim(0, 0.30)
axs[1, 1].set_title("Position Scatter")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")

# Overlay plot
axs[1, 2].plot(t, T_norm, linewidth = 3, label='Temp')
axs[1, 2].plot(t, rho_norm, linewidth = 3, label='Density')
axs[1, 2].plot(t, h_norm, linewidth = 3, label='h')
axs[1, 2].plot(t, vr_norm, linewidth = 3, label='Velocity')
axs[1, 2].set_title("Overlay of Normalized Quantities")
axs[1, 2].set_xlabel("t")
axs[1, 2].set_ylabel("Normalized Values")
axs[1, 2].legend()

# Adjust layout
plt.tight_layout()

plt.savefig('figure.png')

# Show plot
plt.show()







