
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time

unitMass_g = 1.9889199999999997e+35
unitRadius_cm = 3.086e+19
grav_constant_cgs = 6.6738e-08
unitDensity_cgs = unitMass_g / unitRadius_cm**3
unit_u_cgs = grav_constant_cgs * unitMass_g / unitRadius_cm


filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_N120_Mach_10/*.csv'))

plt.ion()
fig, ax = plt.subplots(2, figsize = (18, 5))

k = 0

kb = ''

for j in range(0, len(filz), 10):

	df = pd.read_csv(filz[j])
	
	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	u = df['u'].values / 4000 * 10000 # converting to Kelvin !??? 
	nT = np.where(u > 2e5)[0]
	xT = x[nT]
	yT = y[nT]
	zT = z[nT]
	
	rho = df['rho'].values
	max_rho = np.max(rho)
	nrho = np.where(rho >= 0.8 * max_rho)[0]
	xrho = x[nrho]
	yrho = y[nrho]
	zrho = z[nrho]
	
	ax[0].cla()
	ax[1].cla()
	
	ax[0].set_position([0.30, 0.05, 0.45, 0.85])
	ax[0].scatter(x, y, s = 0.001, color = 'black')
	ax[0].scatter(xT, yT, s = 2.0, color = 'red')
	ax[0].scatter(xrho, yrho, s = 2.0, color = 'blue')
	ax[0].axis(xmin = -2.1, xmax = 2.1)
	ax[0].axis(ymin = -1.1, ymax = 1.1)
	ax[0].set_title('max rho = ' + str(round(max_rho,2))+ '             80% of max rho = ' + str(round(0.8*max_rho,2)))
	
	ax[1].set_position([0.78, 0.05, 0.21, 0.85])
	ax[1].scatter(y, z, s = 0.001, color = 'black')
	ax[1].scatter(yT, zT, s = 2.0, color = 'red')
	ax[1].scatter(xrho, yrho, s = 2.0, color = 'blue')
	ax[1].axis(xmin = -1.1, xmax = 1.1)
	ax[1].axis(ymin = -1.1, ymax = 1.1)
	#ax[1].set_title('80% of max rho = ' + str(round(0.8*max_rho,2)))
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('Temp_rho_analysis.png')



