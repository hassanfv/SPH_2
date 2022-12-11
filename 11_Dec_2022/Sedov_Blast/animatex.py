
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import readchar


with open('PlaneXY.pkl', 'rb') as f:
	nXY = pickle.load(f) # the indices of the particles residing in the XY plane at Z = 0.0

dirx = './Outputs/'
dirx = './Outputs/'

filez = np.sort(os.listdir(dirx))

plt.ion()

fig, ax = plt.subplots(figsize = (6, 5))

for j in range(0, len(filez), 10):

	print(j)

	with open(dirx + filez[j], 'rb') as f:
		dictx = pickle.load(f)

	r = dictx['pos']
	rx = r[:, 0]
	ry = r[:, 1]
	rz = r[:, 2]

	ax.plot(rx[nXY], ry[nXY], marker = 'D', markersize = 2, markeredgecolor = 'black', markerfacecolor = 'white', linestyle = 'None')

	x = y = 0.5

	ax.axis(xmin = -x, xmax = x)
	ax.axis(ymin = -y, ymax = y)
	#ax.axis(ymin = -0.13, ymax = 0)

	fig.canvas.flush_events()
	time.sleep(0.01)
	
	ax.cla()
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break



