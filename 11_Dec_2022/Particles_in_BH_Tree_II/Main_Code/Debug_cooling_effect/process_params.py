
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import imageio
import time
import pandas as pd


df = pd.read_csv('X-954822.csv') #=====> black
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
uAC1 = df['uAC']

df = pd.read_csv('X-948050.csv')
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
uAC2 = df['uAC']


t1 = t1 - 59

xran1 = 55
xran2 = 250

plt.figure(figsize = (18, 12))

plt.subplot(2, 3, 1)
plt.scatter(t1, v1, s = 10, color = 'k')
plt.scatter(t2, v2, s = 5, color = 'b')
plt.title('v vs t')
plt.xlim(xran1, xran2)


plt.subplot(2, 3, 2)
plt.scatter(t1, nH1, s = 10, color = 'k')
plt.scatter(t2, nH2, s = 5, color = 'b')
plt.title('nH vs t')
plt.xlim(xran1, xran2)


plt.subplot(2, 3, 3)
plt.scatter(t1, ut_c1, s = 10, color = 'k')
plt.scatter(t2, ut_c2, s = 5, color = 'b')
plt.title('ut_c vs t')
plt.xlim(xran1, xran2)
for tmp in ut_c1:
  print(f'{tmp:.4f}')


plt.subplot(2, 3, 4)
plt.scatter(t1, ut1, s = 10, color = 'k', label = 'ut')
#plt.scatter(t1, np.abs(ut_c1), s = 10, color = 'grey', label = 'ut_c')
plt.scatter(t2, ut2, s = 5, color = 'b', label = 'ut')
#plt.scatter(t2, np.abs(ut_c2), s = 5, color = 'purple', label = 'ut_c')
plt.title('ut vs t')
plt.xlim(xran1, xran2)
plt.legend()


plt.subplot(2, 3, 5)
plt.scatter(t1, uAad1, s = 10, color = 'k')
plt.scatter(t2, uAad2, s = 5, color = 'b')
plt.title('uAad vs t')
plt.yscale('log')
plt.xlim(xran1, xran2)

if False:
  plt.subplot(2, 3, 6)
  plt.scatter(nH1, ut_c1, s = 10, color = 'k')
  plt.scatter(nH2, ut_c2, s = 5, color = 'b')
  plt.title('ut_c vs nH')

plt.show()





