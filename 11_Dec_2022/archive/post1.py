
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar


filez = np.sort(glob.glob('./Output/*.pkl'))

for j in range(0, len(filez), 5):

	with open(filez[j], 'rb') as f:
		df = pickle.load(f)
	
	r = df['r']
	P = df['P']
	t = df['t']
	rho = df['rho']
	
	plt.cla()
	
	plt.scatter(r, rho, s = 1, color = 'black')
	plt.xlim(-0.4, 0.4)
	plt.ylim(0.0, 1.1)
	plt.title('t = ' + str(np.round(t, 3)))
	plt.draw()
	plt.pause(0.001)
	
	kb = readchar.readkey()
	
	if kb == 'q':
		break


