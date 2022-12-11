
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


unitTime_in_Myr =  0.0341 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 5e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


filz = np.sort(glob.glob('./Outputs/*.pkl'))

#j = -1

plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

kb = ''

for j in range(7000, len(filz), 1):

	print('j = ', j)

	with open(filz[j], 'rb') as f:
		data = pickle.load(f)


	r = data['pos']
	h = data['h']
	print(r.shape)
	
	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	
	print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
	
	ax.cla()

	ax.scatter(x, y, s = 0.02, color = 'black')
	xyrange = 1.0
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)

	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_Myr,4)) + '       t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







