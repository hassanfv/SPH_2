
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
import pandas as pd


unitTime_in_Myr = 0.03066 # Myr


UnitDensity_in_cgs = 1.6e-17

filz = np.sort(glob.glob('./eps_0.001_50k/*.csv'))

plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

kb = ''

for j in range(0, len(filz), 1):

	print('j = ', j)

	df = pd.read_csv(filz[j])

	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	h = df['h'].values
	rho = df['rho'].values
	
	t = float(filz[j].split('/')[-1][2:-4])
	
	print('h = ', np.sort(h))
	print()	
	print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
	
	ax.cla()

	ax.scatter(x, y, s = 0.01, color = 'black')
	xyrange = 0.2
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)

	
	#ax.set_title('t/t_ff = ' + str(np.round(t*unitTime_in_Myr/t_ff_in_Myrs,4)) + '       t_code = ' + str(round(t, 4)))
	ax.set_title('t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







