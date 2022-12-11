
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 9.2e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


filz = np.sort(glob.glob('./Outputs/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

kb = ''

for j in range(600, len(filz), 1):

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
	unitTime_in_kyr = data['unitTime_in_kyr']
	
	print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
	
	ax.cla()

	ax.scatter(x, y, s = 0.1, color = 'black')
	xyrange = 1.0
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)

	
	ax.set_title('t = ' + str(np.round(t*unitTime_in_kyr,2)) + '       t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







