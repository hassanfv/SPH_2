
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  3.5174499013053913 # Myr


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_alpha_0.5/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_alpha_1.0/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_alpha_2.0/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (8, 6))

kb = ''

for j in range(0, len(filz), 10):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']
	
	print('h = ', np.sort(h))
	
	#print(r.shape)

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	#print('rho = ', np.sort(rho))
	
	ax.cla()

	ax.scatter(y, z, s = 0.01, color = 'black')
	xyrange = 1.0
	
	#ax.axis(xmin = -1.2, xmax = 3.2)
	#ax.axis(ymin = -1.2, ymax = 1.5)
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)
	
	#ax.axis(xmin = +1.00, xmax = 1.20)
	#ax.axis(ymin = +0.05, ymax = 0.28)
	
	
	
	#ax.axhline(y = -0.5, linestyle = '--', color = 'blue')
	#ax.axhline(y =  0.5, linestyle = '--', color = 'blue')
	
	#ax.axvline(x = -0.5, linestyle = '--', color = 'blue')
	#ax.axvline(x =  0.5, linestyle = '--', color = 'blue')
	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_Myr,2) > 20000:
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







