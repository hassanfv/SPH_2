
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_csv('GPUout_1600_100k.csv', names = ['x', 'y', 'z'])

nam = 'G-1.210094.csv'
#df = pd.read_csv(nam)

df = pd.read_csv('./200k_no_min_dt_set/' + nam)

rho = df['rho'].values
h = df['h'].values

UnitMass_in_g =  1.98892e+33
UnitRadius_in_cm =  9.2e+16
unitVelocity =  37984.05600294395
UnitDensity_in_cgs =  2.55419372071998e-18
Unit_P_in_cgs =  3.685161353679455e-09
unitTime_in_Myr =  0.07675073137699442
unitTime_in_kyr = unitTime_in_Myr * 1000


t = float(nam[2:-4])

print('sorted h = ', np.sort(h))
print()
print('sorted rho = ', np.sort(rho)*UnitDensity_in_cgs)
print()
print(f'current time in kyr = {(t * unitTime_in_kyr):.3f}')

x = [1, 0] # x-y plane

x = [0, 1] # y-z plane

if x[0]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['x'].values, df['y'].values, s = 0.001, color = 'k')
	xyrange = 1.0
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)


if x[1]:
	plt.figure(figsize = (6, 5))
	plt.scatter(df['y'].values, df['z'].values, s = 0.001, color = 'k')
	xyrange = 1.0
	plt.xlim(-xyrange, xyrange)
	plt.ylim(-xyrange, xyrange)



plt.show()
