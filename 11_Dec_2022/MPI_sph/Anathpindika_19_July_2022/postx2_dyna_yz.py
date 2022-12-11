
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  3.5174499013053913 # Myr


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_Model_4_15k_Tps_1382/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (10, 6))

kb = ''

for j in range(0, len(filz), 1):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']
	
	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	
	#--- defining the x-y range for the plot ----
	#xfrac = 0.1
	#yfrac = 0.1
	cff = 0.4

	n_max_rho = np.where(rho == np.max(rho))[0]
	zcen = z[n_max_rho]
	ycen = y[n_max_rho]
	zlen = np.max(z) - np.min(z)
	ylen = np.max(y) - np.min(y)
	xmin = zcen - cff #xfrac * xlen
	xmax = zcen + cff #xfrac * xlen
	ymin = ycen - cff #yfrac * ylen
	ymax = ycen + cff #yfrac * ylen
	#--------------------------------------------
	
	ax.cla()

	ax.scatter(y, z, s = 0.01, color = 'black')
	xyrange = 1.2
	
	ax.axis(xmin = xmin, xmax = xmax)
	ax.axis(ymin = ymin, ymax = ymax)
	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_Myr,2) > 20000:
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







