
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  12.430967917168827 # Myr


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_9_May/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (10, 6))

kb = ''

for j in range(0, len(filz), 20):

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

	ax.scatter(x, y, s = 0.01, color = 'black')
	xyrange = 1.2
	
	ax.axis(xmin = -1.2, xmax = 3.2)
	ax.axis(ymin = -1.2, ymax = 1.5)
	
	#ax.axis(xmin = 0.9, xmax = 1.3)
	#ax.axis(ymin = -0.15, ymax = 0.2)
	
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







