
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time
import os


filz = np.sort(glob.glob('./Outputs/*.csv'))

res = []

jj = 2000 # 14564

for j in range(0, len(filz), 10):

	df = pd.read_csv(filz[j])

	t = float(filz[j].split('/')[-1][2:-4])

	rho = df['rho'].values
	u = df['u'].values
	
	res.append([t, u[jj], rho[jj]])

res = np.array(res)

t = res[:, 0]
u = res[:, 1]
rho = res[:, 2]


print('The u vs t is created !!!')
print('Press any key to start ....')


plt.ion()
fig, ax = plt.subplots(3, figsize = (18, 5))

k = 0

kb = ''

for j in range(0, len(filz), 10):

	df = pd.read_csv(filz[j])
	
	x = df['x'].values
	y = df['y'].values
	z = df['z'].values
	
	ax[0].cla()
	ax[1].cla()
	ax[2].cla()
	
	ax[0].set_position([0.05, 0.05, 0.20, 0.85])
	ax[0].scatter(t, u/max(u), s = 10, color = 'orange')
	ax[0].scatter(t, rho/max(rho), s = 10, color = 'blue')
	ax[0].scatter(t[k], u[k]/max(u), s = 60, color = 'red')
	ax[0].scatter(t[k], rho[k]/max(rho), s = 60, color = 'purple')
	
	ax[0].set_title('place holder')
		
	k += 1
	
	dxy = 0.6
	
	ax[1].set_position([0.30, 0.05, 0.45, 0.85])
	ax[1].scatter(x, y, s = 0.05, color = 'black')
	ax[1].scatter(x[jj], y[jj], s = 30, facecolor = 'none', edgecolor = 'red')
	ax[1].axis(xmin = -2.1, xmax = 2.1)
	ax[1].axis(ymin = -1.1, ymax = 1.1)
	
	ax[1].axis(xmin = x[jj] - dxy, xmax = x[jj] + dxy)
	ax[1].axis(ymin = y[jj] - dxy, ymax = y[jj] + dxy)



	ax[2].set_position([0.78, 0.05, 0.21, 0.85])
	ax[2].scatter(y, z, s = 0.05, color = 'black')
	ax[2].scatter(y[jj], z[jj], s = 30, facecolor = 'none', edgecolor = 'red')
	#ax[2].axis(xmin = -1.1, xmax = 1.1)
	#ax[2].axis(ymin = -1.1, ymax = 1.1)
	
	ax[2].axis(xmin = y[jj] - dxy, xmax = y[jj] + dxy)
	ax[2].axis(ymin = z[jj] - dxy, ymax = z[jj] + dxy)
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')



