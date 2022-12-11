
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_yr =  15745.497


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('./Outputs_No_AV/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (8, 7))

kb = ''

for j in range(0, len(filz), 20):

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	
	ax.cla()

	ax.scatter(x, y, s = 0.08, color = 'black')
	xyrange = 1.2
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)
	
	#ax.axhline(y = -0.5, linestyle = '--', color = 'blue')
	#ax.axhline(y =  0.5, linestyle = '--', color = 'blue')
	
	#ax.axvline(x = -0.5, linestyle = '--', color = 'blue')
	#ax.axvline(x =  0.5, linestyle = '--', color = 'blue')
	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_yr,2)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_yr,2) > 20000:
	#	kb =readchar.readkey()
	
	if kb == 'q':
		break








