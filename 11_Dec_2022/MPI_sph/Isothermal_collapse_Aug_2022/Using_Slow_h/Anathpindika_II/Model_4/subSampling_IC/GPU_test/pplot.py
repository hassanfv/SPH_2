
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

df = pd.read_csv('G-0.338301.csv')

#df = pd.read_csv('./OutGPU/G-0.002000.csv')

rho = df['rho'].values
h = df['h'].values

print(rho.shape)

print('sorted h = ', np.sort(h))
print()
print('sorted rho = ', np.sort(rho))
print()


if 0:
	fig, ax = plt.subplots(figsize = (7, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	plt.xlim(-1.2, 3.2)
	plt.ylim(-1.5, 1.5)


if 1:
	plt.figure(figsize = (7, 6))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)



plt.show()
