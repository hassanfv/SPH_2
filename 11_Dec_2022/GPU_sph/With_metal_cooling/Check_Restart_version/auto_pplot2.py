
# In this version we only plot in the Y-Z plane.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time
import os


filz = np.sort(glob.glob('./Outputs/*.csv'))


res = []

plt.ion()
fig, ax = plt.subplots(1, figsize = (10, 9))

k = 0

kb = ''

Filter = True

for j in range(0, len(filz), 1):

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
	
	T = df['u'].values / 18 * 10000 # converting to Kelvin !??? 
	nT = np.where(T < 10)[0]
	xT = x[nT]
	yT = y[nT]
	zT = z[nT]
		
	
	ax.cla()
	
	if True:
	
		nx = np.where((x >= -0.1) & (x <= 0.1))[0]
		
		x = x[nx]
		y = y[nx]
		z = z[nx]
	
	
	#ax.set_position([0.78, 0.05, 0.21, 0.85])
	ax.scatter(y, z, s = 0.001, color = 'black')
	ax.scatter(yrho, zrho, s = 1.0, color = 'blue')
	ax.scatter(yT, zT, s = 1.0, color = 'red')
	#ax.axis(xmin = -1.1, xmax = 1.1)
	#ax.axis(ymin = -1.1, ymax = 1.1)
	
	dxy = 0.3
	
	ax.axis(xmin = -dxy, xmax = dxy)
	ax.axis(ymin = -dxy, ymax = dxy)
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('3333.png')



