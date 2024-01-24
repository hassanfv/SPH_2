
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

unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!!!

def plot_dut_vs_t(ndx, color):

  fname = 'X-' + str(ndx) + '.csv'
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
  
  plt.scatter(t1, res1, s = 5, color = color, label = 'Max(log(T)) = ' + str(np.round(max(T1), 2)))
  plt.title('Particle index = ' + str(ndx))


plt.figure(figsize = (12, 12))

plt.subplot(2, 2, 1)  # (2 row, 2 columns, first subplot)
plot_dut_vs_t(822354, 'black')
plot_dut_vs_t(820642, 'blue')
plot_dut_vs_t(822307, 'red')
xran = 0.6
plt.ylim(-xran, xran)
plt.legend()


plt.subplot(2, 2, 2)



plt.savefig('fig.png')

plt.show()





