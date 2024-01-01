
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob


filez = np.sort(glob.glob('./Outputs/*.bin'))


unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!
mH = 1.673534e-24


def plot_circle(xcen, ycen, rad):

    theta = np.linspace(0, 2*np.pi, 100)

    x = xcen + rad * np.cos(theta)
    y = ycen + rad * np.sin(theta)

    plt.plot(x, y, color = 'b')



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



jj = 939445   # [874002 875985 877706 877710 939445 940947]

res = []

plt.figure(figsize = (7, 7))

for i, filename in enumerate(filez):
  x, y, z, vx, vy, vz, rho, h, uB, uA, u, mass, Typ, N = readBinaryFile(filename)

  nxBH = np.where(mass == max(mass))[0][0]

  n = np.where(u != 0.0)[0]
  NN = len(n)
  print('non-zero u or NN = ', len(n))
  print('nxBH = ', nxBH)
  print('sort(mass)', np.sort(mass))
  print('filename = ', filename)
  rho = rho[n]
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

  xt = x[nh]
  yt = y[nh]

  
  plt.scatter(xt, yt, s = 5, color = 'k')
  plt.scatter(xj, yj, s = 20, color = 'lime')
  plt.scatter(x[nxBH:NN], y[nxBH:NN], s = 5, color = 'r')
  plot_circle(xj, yj, 2*h[jj])
  
  
  #plt.xlim(xj - 5*h[jj], xj + 5*h[jj])
  #plt.ylim(yj - 5*h[jj], yj + 5*h[jj])
  
  plt.xlim(0, 0.15)
  plt.ylim(0, 0.15)
  
  plt.draw()  # Draw the current figure
  plt.pause(0.1)  # Pause for half a second with event processing
  plt.clf()  # Clear the figure for the next plot
  
  #--- determining which particle (i.e. gas or outflow) are withing the smoothing length
  NGas = len(np.where(nh < nxBH)[0])
  NOutflow = len(np.where(nh > nxBH)[0])

  print(f'N_Gas = {NGas},  N_Outflow = {NOutflow},  Nngb = {NGas + NOutflow},  h = {h[jj]},  rho = {rho[jj]}')
  print()
 



