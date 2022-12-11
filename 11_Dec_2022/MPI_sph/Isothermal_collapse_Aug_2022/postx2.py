
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time


#unitTime_in_Myr = 0.030653623768340316 # Myr  # Moved into the for loop !

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 5.0e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')



filz = np.sort(glob.glob('./Outputs_20k_B/*.pkl'))
filz = np.sort(glob.glob('./Outputs/*.pkl'))



plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

kb = ''

for j in range(300, len(filz), 2):

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
	unitTime_in_Myr = data['unitTime'] / 3600. / 24. / 365.25 / 1.e6 # in Myrs
	t_ff_in_Myrs = data['t_ff'] / 3600. / 24. / 365.25 / 1.e6 # in Myrs
	
	print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
	
	ax.cla()

	ax.scatter(x, y, s = 0.01, color = 'black')
	xyrange = 0.1
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)

	
	ax.set_title('t/t_ff = ' + str(np.round(t*unitTime_in_Myr/t_ff_in_Myrs,4)) + '       t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break

plt.savefig('1111.png')







