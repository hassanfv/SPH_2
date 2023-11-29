

import numpy as np
import matplotlib.pyplot as plt
import pickle
import readchar
import os
import pandas as pd
import time
from numba import njit
import csv
import struct


#loads 1D PPM results
def load_ppm_result():
	gamma = 5./3.
	rost = 3./4./np.pi
	est = 1.054811e-1  / 1.05
	pst = rost*est
	vst = np.sqrt(est)
	rst = 1.257607
	time = 0

	radius = np.zeros(350)
	rho = np.zeros(350)
	vr = np.zeros(350)
	press = np.zeros(350)

	with open('./ppm_profile/ppm1oaf') as csvfile:
		readCSV = csv.reader(csvfile)
		line = 0
		for row in readCSV:
			line = line+1
			values = row[0].split()
			if(line == 1):
				time = values[1]
				continue
			if(line == 352):
				break

			radius[line -2] = float(values[1]) /rst*1e-11
			rho[line -2] = float(values[2]) /rost
			vr[line -2] = float(values[4]) /vst*1e-8
			press[line -2] = float(values[3])/pst*1e-16

	rho=rho*(3.0/(4*np.pi))
	press = press*(3.0/(4*np.pi))

	entropy = press / rho**gamma

	return radius, rho, vr, entropy, press



#bins the particle properties
def get_data_bins(radius,Density, vr):
	number_bins = 50
	min_r = 0.01
	max_r = 1.0

	r_bin = np.zeros(number_bins)
	vr_bin = np.zeros(number_bins)

	rho_bin = np.zeros(number_bins)
	count_bin = np.zeros(number_bins)
	#entropy_bin = np.zeros(number_bins)

	for i in range(radius.size):
		if(radius[i] < min_r or radius[i] > max_r):
			continue
		bin_value = int((np.log10(radius[i] / min_r))/(np.log10(max_r/min_r)) * number_bins)
		count_bin[bin_value] = count_bin[bin_value] + 1
		vr_bin[bin_value] = vr_bin[bin_value] + vr[i]
		rho_bin[bin_value] = rho_bin[bin_value] + Density[i]
		#entropy_bin[bin_value] = entropy_bin[bin_value] + A[i]


	vr_bin /= count_bin
	rho_bin /= count_bin
	#entropy_bin /= count_bin

	for i in range(number_bins):
		r_bin[i] = (i+0.5)* (np.log10(max_r/min_r)/number_bins) + np.log10(min_r)
		r_bin[i] = 10**r_bin[i]
		print(count_bin[i])

	return r_bin,rho_bin, vr_bin #, entropy_bin




def readBinaryFile(filename):
    with open(filename, 'rb') as f:
        # Read N and N_ionFrac
        N, N_ionFrac = struct.unpack('ii', f.read(2 * 4))  # 4 bytes each for two integers

        # Read arrays
        Typ = np.array(struct.unpack(f'{N}i', f.read(N * 4)))
        x = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        y = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        z = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vx = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vy = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        vz = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        rho = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        h = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        u = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        mass = np.array(struct.unpack(f'{N}f', f.read(N * 4)))
        ionFrac = np.array(struct.unpack(f'{N_ionFrac}f', f.read(N_ionFrac * 4)))

    # Return the data
    return N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac 



filename = 'G-0.799968.bin'

radius_ppm, rho_ppm, vr_ppm, entropy, press = load_ppm_result()

# Usage
N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac = readBinaryFile(filename)

print("np.sort(h) = ", np.sort(h))

print('N = ', N)
print('x.shape = ', x.shape)

rx = x
ry = y
rz = z

plt.scatter(y, z, s = 0.1, color = 'k')
plt.show()


vx = vx
vy = vy
vz = vz


gamma = 5./3.

tt = 0.0

rr = np.sqrt(rx**2 + ry**2 + rz**2)

vv = np.sqrt(vx**2 + vy**2 + vz**2)


vv = (rx*vx + ry*vy + rz*vz) / rr


rr, rho, vv = get_data_bins(rr, rho, vv)

dt = 1e-4

with open('vrGadget.pkl', 'rb') as f:
	vrG = pickle.load(f)
rG = vrG['r']
vr = vrG['vr']

m = mass

with open('rhoGadget.pkl', 'rb') as f:
	DensityG = pickle.load(f)
rhoG = DensityG['rho']


#P = (gamma - 1.) * u * rho

with open('PGadget.pkl', 'rb') as f:
	PressureG = pickle.load(f)
PG = PressureG['P']



fig, ax = plt.subplots(3, figsize = (16, 9))

ax[0].cla()
ax[1].cla()
ax[2].cla()

ax[0].set_position([0.05, 0.50, 0.27, 0.40])
ax[0].scatter(rr, vv, s = 1, alpha = 1.0, color = 'k', label='This work')
ax[0].scatter(rG, vr, s = 10, color = 'lime', label = 'Gadget 4')
ax[0].plot(radius_ppm, vr_ppm, color = 'orange', label = 'PPM')
ax[0].text(0.01, 1.4, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[0].axis(xmin = 0.005, xmax = 1.2)
ax[0].axis(ymin = -2, ymax = 2)
ax[0].set_xlabel('Radius')
ax[0].set_ylabel('Velocity')
ax[0].set_xscale('log')
ax[0].legend()


ax[1].set_position([0.36, 0.50, 0.28, 0.40])
ax[1].scatter(rr, rho, s = 12, alpha = 1.0, color = 'k', label='This work')
ax[1].scatter(rG, rhoG, s = 10,  color = 'lime', label = 'Gadget 4')
ax[1].plot(radius_ppm, rho_ppm, color = 'orange', label = 'PPM')
ax[1].text(0.01, 150, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[1].axis(xmin = 0.005, xmax = 1.2)
ax[1].axis(ymin = 0.01, ymax = 400)
ax[1].set_xlabel('Radius')
ax[1].set_ylabel('Density')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].legend()

'''
ax[2].set_position([0.68, 0.50, 0.28, 0.40])
ax[2].scatter(rr, P, s = 12, alpha = 1.0, color = 'k', label='This work')
ax[2].scatter(rG, PG, s = 10, color = 'lime', label = 'Gadget 4')
ax[2].plot(radius_ppm, press, color = 'orange', label = 'PPM')
ax[2].text(0.01, 150, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[2].axis(xmin = 0.005, xmax = 1.2)
ax[2].axis(ymin = 0.01, ymax = 400)
ax[2].set_xlabel('Radius')
ax[2].set_ylabel('Pressure')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].legend()
'''

plt.savefig('SnapShotx.png')

plt.show()




