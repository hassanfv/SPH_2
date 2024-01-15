
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import imageio
import time
import pandas as pd

kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24
gamma = 5./3.

unit_velocity_cgs = 1.34181e+06 # cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!
unit_rho = 2.83261e-24 # !!!!!!!!!!!!!!!!!!!

filez = np.sort(glob.glob('./Outputs_1181X/*.bin'))

#===== readBinaryFile
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


#===== plot_dut_vs_t
def plot_dut_vs_t(ndx, file_ndx, color):

  fname = 'X-' + str(ndx) + '.csv' # Note that this csv file is created by "param_evolution.py"
  df = pd.read_csv(fname)
  t1 = df['t']
  r1 = df['r']
  nH1 = df['nH']
  v1 = df['v']
  P1 = df['P']
  ut1 = df['ut']
  ut_pre1 = df['ut_pre']
  ut_c1 = df['ut_c']
  uBad1 = df['uBad']
  uAad1 = df['uAad']
  uAC1 = df['uAC'].values
  T1 = (gamma - 1) * mH / kB * mu * uAC1 * unit_u
  T1 = np.log10(T1)
  
  N1 = len(t1)
  res1 = np.zeros(N1)
  for i in range(N1-1):
    res1[i] = ut1[i+1] - ut1[i]
  
  plt.scatter(t1, res1, s = 5, color = color, label = 'T = ' + str(np.round(max(T1), 2)) + '   ' + str(ndx))
  plt.scatter(t1[file_ndx], res1[file_ndx], s = 50, color = color)
  


#===== plot_x_vs_y
def plot_x_vs_y(file_ndx, particle_ndx, color, colorx, condition):

  fname = filez[file_ndx]
  x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, Nn = readBinaryFile(fname)

  n = np.where(u != 0.0)[0]
  
  x = x[n]
  y = y[n]
  z = z[n]
  
  nz = np.where(np.abs(z) < 0.02)[0]
  xx = x[nz]
  yy = y[nz]
  zz = z[nz]
  
  if condition:
    plt.scatter(xx, yy, s=0.5, color='k')
  plt.scatter(x[particle_ndx], y[particle_ndx], s=30, color=colorx) # Here we use original x!
  plt.xlim(0.05, 0.2)
  plt.ylim(-0.05, 0.05)
  plt.title('Particle index = ' + str(particle_ndx) + ' (' + color + ')')



# Black
file_ndx_1 = 170#155#170
particle_ndx_1 = 822354

# Blue
file_ndx_2 = 158#151#158
particle_ndx_2 = 820642

# Red
file_ndx_3 = 165#155#165
particle_ndx_3 = 822307


plt.figure(figsize = (12, 12))

color1 = 'black'
color2 = 'blue'
color3 = 'red'

plt.subplot(2, 2, 1)  # (2 row, 2 columns, first subplot)
plot_dut_vs_t(particle_ndx_1, file_ndx_1, color1)
plot_dut_vs_t(particle_ndx_2, file_ndx_2, color2)
plot_dut_vs_t(particle_ndx_3, file_ndx_3, color3)
yran = 0.6
plt.ylim(-yran, yran)
xran1 = 120
xran2 = 250
plt.xlim(xran1, xran2)
plt.legend()


plt.subplot(2, 2, 2)
plot_x_vs_y(file_ndx_1, particle_ndx_1, color1, 'red', True)
plot_x_vs_y(file_ndx_2, particle_ndx_2, color2, 'blue', False)
plot_x_vs_y(file_ndx_3, particle_ndx_3, color3, 'lime', False)

plt.subplot(2, 2, 3)
plot_x_vs_y(file_ndx_2, particle_ndx_2, color2, 'red', True)

plt.subplot(2, 2, 4)
plot_x_vs_y(file_ndx_3, particle_ndx_3, color3, 'red', True)


plt.savefig('fig.png')

plt.show()





