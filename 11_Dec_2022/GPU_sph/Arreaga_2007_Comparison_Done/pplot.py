
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

#df = pd.read_csv('./eps_180k/G-1.590858.csv')

df = pd.read_csv('/mnt/Linux_Shared_Folder_2022/GPU_sph/Arreaga/180k_eps_0.001/G-1.513622.csv')

rho = df['rho'].values
h = df['h'].values

UnitMass_in_g =  1.98892e+33
UnitRadius_in_cm =  4.99e+16
unitVelocity =  51575.681877433715
UnitDensity_in_cgs =  1.6007211309378248e-17
Unit_P_in_cgs =  4.2579997828398464e-08
unitTime_in_Myr =  0.03065854923019242

print('sorted h = ', np.sort(h))
#print('sorted rho = ', np.sort(rho))
print('sorted rho = ', np.sort(rho)*UnitDensity_in_cgs)
print()

x = [1, 0] # x-y plane

#x = [0, 1] # y-z plane

if x[0]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.0002, color = 'k')
	xyrange = 0.2
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)
	plt.axvline(x = -0.0202, linestyle = '--')


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	xyrange = 1.0
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)



plt.show()
