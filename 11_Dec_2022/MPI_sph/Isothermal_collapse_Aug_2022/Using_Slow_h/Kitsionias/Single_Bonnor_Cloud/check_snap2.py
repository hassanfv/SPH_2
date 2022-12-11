
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import readchar

M_sun = 1.98892e33
UnitMass_in_g = 75. * M_sun

UnitRadius_in_cm = 1.7877e19
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

filz = np.sort(glob.glob('./output_Gad_BE/*.hdf5'))


plt.ion()
fig, ax = plt.subplots(figsize = (7, 6))

kb = ''

for j in range(len(filz)):

	print(j)

	file = h5py.File(filz[j], 'r')
	# ['Coordinates', 'Density', 'InternalEnergy', 'Masses', 'ParticleIDs', 'SmoothingLength', 'Velocities']

	coord = file['PartType0']['Coordinates']

	rho = np.array(file['PartType0']['Density']) * UnitDensity_in_cgs
	print('rho = ', np.sort(rho))


	h = file['PartType0']['SmoothingLength']
	print('h = ', np.sort(h))
	
	Velocities = file['PartType0']['Velocities']
	VX = np.abs(Velocities[:, 0])
	print('VX = ', np.sort(np.abs(Velocities[:, 0])))
	
	print()
	
	nVx = np.where(VX < 0.5)[0]	
	
	
	ax.cla()

	ax.scatter(coord[:, 0], coord[:, 1], s = 0.02, color = 'black')
	#ax.scatter(coord[nVx, 0], coord[nVx, 1], s = 0.1, color = 'red')
	
	xyrange = 1.5
	
	ax.axis(xmin = -xyrange, xmax = xyrange)
	ax.axis(ymin = -xyrange, ymax = xyrange)
	
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break




