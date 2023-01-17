import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('testCool.csv')

u = df['u'].values
heat = df['heating'].values
cool = df['cooling'].values

kB = 1.3807e-16  # cm2 g s-2 K-1
mH = 1.6726e-24 # gram
gamma = 5.0/3.0
muT = 1.22

utmp = 1e13

T = (gamma - 1) * mH * muT / kB * utmp

print('T = ', T)

nn = np.argmin(np.abs(u - utmp))

print('heat - cool at T = ', heat[nn] - cool[nn])


plt.scatter(np.log10(u), np.log10(cool), s = 10, color = 'blue')
plt.scatter(np.log10(u), np.log10(heat), s = 10, color = 'red')

plt.show()




