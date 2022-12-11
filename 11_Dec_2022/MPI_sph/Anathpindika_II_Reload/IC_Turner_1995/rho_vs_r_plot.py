
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import readchar


filez = np.sort(glob.glob('./Outputs/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (8, 6))


for j in range(len(filez)):

	with open(filez[j], 'rb') as f:
		data = pickle.load(f)
	
	r = data['pos']
	rho = data['rho']
	h = data['h']
	
	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]

	rr = (x*x + y*y + z*z)**0.5

	ax.cla()
	#ax.scatter(rr, rho, s = 1, color = 'black')
	ax.scatter(x, y, s = 1, color = 'k')
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('fig.png')
	
	




