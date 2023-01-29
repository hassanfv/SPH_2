
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time
import os

unitMass_g = 1.9889199999999997e+35
unitRadius_cm = 3.086e+19
grav_constant_cgs = 6.6738e-08
unitDensity_cgs = unitMass_g / unitRadius_cm**3
unit_u_cgs = grav_constant_cgs * unitMass_g / unitRadius_cm

print(f'unitDensity_cgs = {unitDensity_cgs:.4E}')

filz = np.sort(glob.glob('./Outputs/*.csv'))


res = []

plt.ion()
fig, ax = plt.subplots(2, figsize = (18, 5))

k = 0

kb = ''

Filter = True

for j in range(0, len(filz), 50):

	df = pd.read_csv(filz[j])
	
	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	rho = df['rho'].values
	max_rho = np.max(rho)
	nrho = np.where(rho >= 30.0)[0]
	xrho = x[nrho]
	yrho = y[nrho]
	zrho = z[nrho]
	
	T = df['u'].values / 200 * 10000 # converting to Kelvin !??? 
	nT = np.where((T > 400) & (T < 500))[0]
	xT = x[nT]
	yT = y[nT]
	zT = z[nT]
		
	
	ax[0].cla()
	ax[0].cla()
	
	ax[0].set_position([0.30, 0.05, 0.45, 0.85])
	ax[0].scatter(x, y, s = 0.001, color = 'black')
	ax[0].scatter(xT, yT, s = 1.0, color = 'red')
	ax[0].axis(xmin = -2.1, xmax = 2.1)
	ax[0].axis(ymin = -1.1, ymax = 1.1)
	
	
	if Filter:
	
		nx = np.where((x >= -0.1) & (x <= 0.1))[0]
		
		x = x[nx]
		y = y[nx]
		z = z[nx]
	
	
	ax[1].set_position([0.78, 0.05, 0.21, 0.85])
	ax[1].scatter(y, z, s = 0.00001, color = 'black')
	ax[1].scatter(yrho, zrho, s = 0.01, color = 'blue')
	ax[1].scatter(yT, zT, s = 1.0, color = 'red')
	ax[1].axis(xmin = -1.1, xmax = 1.1)
	ax[1].axis(ymin = -1.1, ymax = 1.1)
	
	#ax[1].axis(xmin = -0.3, xmax = 0.3)
	#ax[1].axis(ymin = -0.65, ymax = -0.0)
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('2222.png')



