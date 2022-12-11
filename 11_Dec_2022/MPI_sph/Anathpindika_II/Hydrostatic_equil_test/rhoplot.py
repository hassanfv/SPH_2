
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  2.108823835395 # Myr


filz = np.sort(glob.glob('./Outputs/*.pkl'))
#filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_Model_3_Anathpindika_2009/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (6, 6))

kb = ''

for j in range(0, len(filz), 10):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	
	rr = (x**2 + y**2 + z**2)**0.5
	
	ax.cla()

	ax.scatter(rr, rho, s = 0.1, color = 'black')

	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	#if np.round(t*unitTime_in_Myr,2) > 20000:
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







