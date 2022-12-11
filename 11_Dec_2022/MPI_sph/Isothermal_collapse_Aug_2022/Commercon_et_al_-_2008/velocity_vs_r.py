#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import time
import readchar

M_sun = 1.98992e+33 # gram
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!

G = 6.6738e-8
rgas = 7.07e16                                     # The initial radius of the cloud in cm
rho0 = 1.35e-18                                   # The initial average density
tff = np.sqrt(3*np.pi/(32*G*rho0))                # The free-fall time = 3.4e4 yr
unitTime_in_s = tff                               # Scaling time to free-fall time
unitLength_in_cm = rgas                           # Scaling distance to the initial cloud radius

UnitDensity_in_cgs = UnitMass_in_g / unitLength_in_cm**3

print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs} g/cm^3')

unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s          # The internal velocity unit
print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')

filz = np.sort(glob.glob('./Outputs_a_0.5_B_0.04_mine_Exact_h/*.pkl'))


plt.ion()
fig, ax = plt.subplots(figsize = (6, 6))
kb = ''

for j in range(0, len(filz), 20):


	with open(filz[j], 'rb') as f:
	    data = pickle.load(f)

	r = data['pos']
	h = data['h']
	v = data['v']
	unitTime_in_kyr = data['unitTime_in_kyr']
	print(r.shape)

	print('h = ', np.sort(h))

	x = r[:, 0]
	y = r[:, 1]
	z = r[:, 2]
	t = data['current_t']
	rho = data['rho']
	unitTime_in_kyr = data['unitTime_in_kyr']

	print('rho = ', np.sort(rho)*UnitDensity_in_cgs)


	# ## Selecting particles in a thin shell parallel to the x-y plane
	delta = 0.02

	nz = np.where((z >= -delta) & (z <= delta))[0]
	print(len(nz))

	rx = x[nz]
	ry = y[nz]
	rz = z[nz]

	vt = v[nz]

	radius = (rx*rx + ry*ry + rz*rz)**0.5

	vr = (vt[:, 0]*rx + vt[:, 1]*ry + vt[:, 2]*rz)/radius

	logR = np.log10(radius)
	rgrid = np.logspace(min(logR), max(logR), 40)

	res = []

	for i in range(len(rgrid)-1):
	    
	    nx = np.where((radius > rgrid[i]) & (radius <= rgrid[i+1]))[0]
	    
	    res.append([rgrid[i], np.mean(vr[nx])])

	res = np.array(res)

	R = res[:, 0] * unitLength_in_cm
	vr = res[:, 1] * unitVelocity_in_cm_per_s

	ax.cla()

	ax.scatter(np.log10(R), vr, s = 5, color = 'k')
	ax.axis(xmin = 13.5, xmax = 17)
	ax.set_title('t = ' + str(np.round(t*unitTime_in_kyr,2)) + '       t_code = ' + str(round(t, 4)))
	fig.canvas.flush_events()
	time.sleep(0.01)
	
	kb =readchar.readkey()
	
	if kb == 'q':
		break



