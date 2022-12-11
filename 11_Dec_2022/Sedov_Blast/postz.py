
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


with open('PlaneXY.pkl', 'rb') as f:
	nXY = pickle.load(f) # the indices of the particles residing in the XY plane at Z = 0.0


j = 400


#dirx = './Outputs/'
dirx = './Outputs/'



filez = np.sort(os.listdir(dirx))
with open(dirx + filez[j], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]

t = dictx['current_t']

plt.figure(figsize = (6, 5))

plt.plot(rx[nXY], ry[nXY], marker = 'D', markersize = 2, markeredgecolor = 'black', markerfacecolor = 'white', linestyle = 'None')

x = y = 0.5

plt.xlim(-x, x)
plt.ylim(-y, y)

plt.title('t = ' + str(round(t, 3)))

plt.show()




