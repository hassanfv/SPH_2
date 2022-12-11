
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import readchar
import time
import h5py


#unitTime_in_Myr =  0.07673840095663824 # Myr

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 4.95e+16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')


filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/output_24k_Gad/*.hdf5'))


plt.ion()
fig, ax = plt.subplots(figsize = (6, 5))

kb = ''

for j in range(1200, len(filz), 1):  # 35.54 + 1 = 36.54

	filename = '/mnt/Linux_Shared_Folder_2022/output_24k_Gad/snap_' + str(j).zfill(4) + '.hdf5'

	print('j = ', j)

	file = h5py.File(filename, 'r')


	r = file['PartType0']['Coordinates']
	h = file['PartType0']['SmoothingLength']
	print(r.shape)
	
	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	rho = list(file['PartType0']['Density'])
	
	rho = np.sort(rho)*UnitDensity_in_cgs
	print('rho = ', rho)
	
	nrho = np.sum(rho >= 1e-13)
	
	print('N_core = ', nrho) # See Commercon et al (2008).
	
	ax.cla()

	ax.scatter(x, y, s = 0.01, color = 'black')
	xyrange = 0.08
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)

	
	#ax.set_title('t = ' + str(np.round(t*unitTime_in_kyr,2)) + '       t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	print()
	
	if kb == 'q':
		break

plt.savefig('1111G.png')







