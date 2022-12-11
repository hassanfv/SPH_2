
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

df = pd.read_csv('G-0.074500.csv')

#df = pd.read_csv('./OutGPU_35k/G-0.371927.csv')

rho = df['rho'].values
h = df['h'].values

print('sorted rho = ', np.sort(rho))
print('sorted h = ', np.sort(h))
print()


if 0:
	plt.figure(figsize = (6, 4))
	plt.scatter(df['x'].values, df['y'].values, s = 0.01, color = 'k')
	plt.xlim(-1.2, 3.2)
	plt.ylim(-1.5, 1.5)


if 1:
	plt.figure(figsize = (7, 6))
	plt.scatter(df['y'].values, df['z'].values, s = 0.01, color = 'k')
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)



plt.show()
