
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



df = pd.read_csv('X-822354.csv') #======> black
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


df = pd.read_csv('X-820642.csv')  #========> blue
t2 = df['t']
r2 = df['r']
nH2 = df['nH']
v2 = df['v']
P2 = df['P']
ut2 = df['ut']
ut_pre2 = df['ut_pre']
ut_c2 = df['ut_c']
uBad2 = df['uBad']
uAad2 = df['uAad']
uAC2 = df['uAC'].values
T2 = (gamma - 1) * mH / kB * mu * uAC2 * unit_u
T2 = np.log10(T2)



df = pd.read_csv('X-822307.csv')  #========> red
t3 = df['t']
r3 = df['r']
nH3 = df['nH']
v3 = df['v']
P3 = df['P']
ut3 = df['ut']
ut_pre3 = df['ut_pre']
ut_c3 = df['ut_c']
uBad3 = df['uBad']
uAad3 = df['uAad']
uAC3 = df['uAC'].values
T3 = (gamma - 1) * mH / kB * mu * uAC3 * unit_u
T3 = np.log10(T3)


N1 = len(t1)
res1 = np.zeros(N1)
for i in range(N1-1):
  res1[i] = ut1[i+1] - ut1[i]


N2 = len(t2)
res2 = np.zeros(N2)
for i in range(N2-1):
  res2[i] = ut2[i+1] - ut2[i]


N3 = len(t3)
res3 = np.zeros(N3)
for i in range(N3-1):
  res3[i] = ut3[i+1] - ut3[i]


plt.figure(figsize = (10, 7))

plt.scatter(t1, res1, s = 10, color = 'k', label = 'Max(log(T)) = ' + str(np.round(max(T1), 2)))
plt.scatter(t2, res2, s = 10, color = 'b', label = 'Max(log(T)) = ' + str(np.round(max(T2), 2)))
plt.scatter(t3, res3, s = 10, color = 'r', label = 'Max(log(T)) = ' + str(np.round(max(T3), 2)))

xran = 0.6

plt.ylim(-xran, xran)
plt.legend()

plt.savefig('fig.png')

plt.show()





