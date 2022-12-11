
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

#df = pd.read_csv('G-0.766121.csv')

df = pd.read_csv('./600k_hfv_2/G-0.766121.csv')

rho = df['rho'].values
h = df['h'].values

UnitMass_in_g =  1.9890000000000002e+34
UnitRadius_in_cm =  6.958465397888282e+17
unitVelocity =  43672.48212128877
UnitDensity_in_cgs =  5.903293380000827e-20
Unit_P_in_cgs =  1.1259267014904865e-10

print('sorted h = ', np.sort(h))
print('sorted rho = ', np.sort(rho))
print('sorted rho = ', np.sort(rho)*UnitDensity_in_cgs)
print()

x = [1, 0] # x-y plane

x = [0, 1] # y-z plane

if x[0]:
	plt.figure(figsize = (6, 4))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	plt.xlim(-1.2, 3.2)
	plt.ylim(-1.5, 1.5)


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	xyrange = 0.1
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)
	
	#.xlim(0.025, 0.04)
	#plt.ylim(-0.025, 0.015)
	
	#plt.xlim(0.01, 0.050)
	#plt.ylim(-0.03, 0.025)



plt.show()


