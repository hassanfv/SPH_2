
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

df = pd.read_csv('./Outputs/G-0.002590.csv')

#df = pd.read_csv('/mnt/Linux_Shared_Folder_2022/GPU_sph/Arreaga/180k_eps_0.001/G-1.513622.csv')

rho = df['rho'].values
h = df['h'].values


x = [1, 0] # x-y plane

#x = [0, 1] # y-z plane

if x[0]:
	plt.figure(figsize = (11, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	#plt.xlim(-2.2, 2.2)
	#plt.ylim(-1, 1)
	#plt.axvline(x = -0.0202, linestyle = '--')


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	xyrange = 1.0
	#plt.xlim(-xyrange, xyrange)
	#plt.ylim(-xyrange, xyrange)



plt.show()
