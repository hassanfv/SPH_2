
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
def plot_dut_vs_t(ndx, file_ndx, color): # ndx is particle index.

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
  
  #plt.scatter(t1, res1, s = 5, color = color, label = 'T = ' + str(np.round(max(T1), 2)) + '   ' + str(ndx))
  #plt.scatter(t1[file_ndx], res1[file_ndx], s = 50, color = color)
  plt.scatter(t1, T1, s = 5, color = color)
  #plt.scatter(t1, nH1, s = 1, color = color)
  #plt.scatter(t1[file_ndx], nH1[file_ndx], s = 120, color = color)
  plt.scatter(t1[file_ndx], T1[file_ndx], s = 120, color = color)
  


#===== plot_x_vs_y
def plot_x_vs_y(file_ndx, particle_ndx, color, colorx, condition, coeff):

  fname = filez[file_ndx]
  x, y, z, vx, vy, vz, rho, h, dudt, dudt_pre, uBad, uAad, u, mass, Typ, Nn = readBinaryFile(fname)

  n = np.where(u != 0.0)[0]
  
  x = x[n]
  y = y[n]
  z = z[n]
  
  h = h[n]
  
  nz = np.where(np.abs(z) < 0.01)[0]
  xx = x[nz]
  yy = y[nz]
  zz = z[nz]
  
  if condition:
    plt.scatter(xx, yy, s = 1, color='k')
    
  plt.scatter(x[particle_ndx], y[particle_ndx], s=50, color=colorx) # Here we use original x!
  
  stp = 0.025
  #if condition:
  #  plt.xlim(x[particle_ndx] - stp, x[particle_ndx] + stp)
  #  plt.ylim(y[particle_ndx] - stp, y[particle_ndx] + stp)
  
  
  #plt.text(x[particle_ndx] - stp+stp/6, y[particle_ndx] - stp + coeff*stp/4, 'h = ' + str(h[particle_ndx]), color = colorx, size = 20)
  
  plt.xlim(0.04, 0.22)
  plt.ylim(-0.06, 0.06)
  #plt.title('Particle index = ' + str(particle_ndx) + ' (' + color + ')')
  plt.xlabel('x')
  plt.ylabel('y')




output_folder = './Check_plots/'


plt.figure(figsize = (18, 8))

for file_ndx_2 in range(80, len(filez)):

  # lime
  file_ndx_1 = 170
  particle_ndx_1 = 751782

  # Blue
  #file_ndx_2 = 158
  particle_ndx_2 = 751923

  # Red
  file_ndx_3 = 165
  particle_ndx_3 = 751969

  color1 = 'lime'
  color2 = 'blue'
  color3 = 'red'

  plt.clf()

  plt.subplot(1, 2, 1)  # (2 row, 2 columns, first subplot)
  plot_dut_vs_t(particle_ndx_1, file_ndx_2, color1)
  plot_dut_vs_t(particle_ndx_2, file_ndx_2, color2)
  plot_dut_vs_t(particle_ndx_3, file_ndx_2, color3)
  yran = 0.6
  #plt.ylim(-yran, yran)
  xran1 = 80
  xran2 = 300
  plt.xlim(xran1, xran2)
  plt.xlabel('Time')
  plt.ylabel('log(Temperature)')


  plt.subplot(1, 2, 2)
  plot_x_vs_y(file_ndx_2, particle_ndx_1, color1, 'lime', True, 1.0)
  plot_x_vs_y(file_ndx_2, particle_ndx_2, color2, 'blue', False,2.0)
  plot_x_vs_y(file_ndx_2, particle_ndx_3, color3, 'red', False, 3.0)
  
  filename = f'{output_folder}plot_{file_ndx_2}.png'  # Filename for the plot
  plt.savefig(filename)
  
  plt.draw()
  plt.pause(0.1)


plt.savefig('fig.png')

plt.show()





