
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time


filz = np.sort(glob.glob('./Outputs/*.csv'))

res = []

jj = 135165 #13500

for nam in filz:

	df = pd.read_csv(nam)

	t = float(nam.split('/')[-1][2:-4])

	rho = df['rho'].values
	u = df['u'].values
	
	res.append([t, u[jj]])

res = np.array(res)

print(res)

t = res[:, 0]
u = res[:, 1]

print('The u vs t is created !!!')


plt.ion()
fig, ax = plt.subplots(3, figsize = (18, 5))

k = 0

for j in range(len(filz)):

	df = pd.read_csv(filz[j])
	
	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	ax[0].cla()
	ax[1].cla()
	ax[2].cla()
	
	ax[0].set_position([0.05, 0.05, 0.20, 0.95])
	ax[0].scatter(t, u, s = 10, color = 'black')
	ax[0].scatter(t[k], u[k], s = 60, color = 'red')
	k += 1
	
	ax[1].set_position([0.30, 0.05, 0.45, 0.95])
	ax[1].scatter(x, y, s = 0.001, color = 'black')
	ax[1].scatter(x[jj], y[jj], s = 60, color = 'red')
	ax[1].axis(xmin = -2.1, xmax = 2.1)
	ax[1].axis(ymin = -1.1, ymax = 1.1)
	
	ax[2].set_position([0.78, 0.05, 0.21, 0.95])
	ax[2].scatter(y, z, s = 0.001, color = 'black')
	ax[2].scatter(y[jj], z[jj], s = 30, color = 'red')
	ax[2].axis(xmin = -1.1, xmax = 1.1)
	ax[2].axis(ymin = -1.1, ymax = 1.1)
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')



