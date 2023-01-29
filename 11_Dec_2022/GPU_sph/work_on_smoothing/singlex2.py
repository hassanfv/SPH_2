

import numpy as np
import matplotlib.pyplot as plt
import pickle
import readchar
import os
import pandas as pd
import time
from numba import njit
import csv


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


#===== smooth_hX (non-parallel)
@njit
def do_smoothingX(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j, 0] - subpos[i, 0]
            dy = pos[j, 1] - subpos[i, 1]
            dz = pos[j, 2] - subpos[i, 2]
            dist[j] = (dx**2 + dy**2 + dz**2)**0.5

        hres.append(np.sort(dist)[50])

    return np.array(hres) * 0.5


#===== getDensityx
@njit
def getDensity(pos, m, h):  # You may want to change your Kernel !!

	N = pos.shape[0]
	rho = np.zeros(N)

	for i in range(N):

		s = 0.0

		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = np.sqrt(dx**2 + dy**2 + dz**2)
			
			hij = 0.5 * (h[i] + h[j])

			sig = 1.0/np.pi
			q = rr / hij

			WIij = 0.0

			if q <= 1.0:
				WIij = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

			if (q > 1.0) and (q <= 2.0):
				WIij = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3
				
			s += m[j] * WIij

		rho[i] = s

	return rho


radius_ppm, rho_ppm, vr_ppm, entropy, press = load_ppm_result()


dirx = './Outputs/'


filez = np.sort(os.listdir(dirx))

fig, ax = plt.subplots(3, figsize = (16, 9))

kb = ''
res = []

gamma = 5./3.



j = 235

with open(dirx + filez[j], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]

#plt.scatter(rx, ry, s = 1)
#plt.show()

tt = dictx['current_t']

rr = np.sqrt(rx**2 + ry**2 + rz**2)
	
v = dictx['v']
vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

vv = np.sqrt(vx**2 + vy**2 + vz**2)

tarr = np.zeros(r.shape[0])

#for xt, yt, zt, vxt, vyt, vzt in zip(rx, ry, rz, vx, vy, vz):
for i in range(r.shape[0]):
	
	#if (xt < 0.0) & (yt < 0.0) & (vxt > 0.0) & (vyt > 0.0) & (zt > 0.0) & (vzt < 0.0):
	if (rx[i] < 0.0) & (ry[i] < 0.0) & (vx[i] > 0.0) & (vy[i] > 0.0) & (rz[i] > 0.0) & (vz[i] < 0.0):
		vv[i] = -vv[i]
	if (rx[i] < 0.0) & (ry[i] < 0.0) & (vx[i] > 0.0) & (vy[i] > 0.0) & (rz[i] < 0.0) & (vz[i] > 0.0):
		vv[i] = -vv[i]
	
	if (rx[i] > 0.0) & (ry[i] < 0.0) & (vx[i] < 0.0) & (vy[i] > 0.0) & (rz[i] > 0.0) & (vz[i] < 0.0):
		vv[i] = -vv[i]
	if (rx[i] > 0.0) & (ry[i] < 0.0) & (vx[i] < 0.0) & (vy[i] > 0.0) & (rz[i] < 0.0) & (vz[i] > 0.0):
		vv[i] = -vv[i]
	
	if (rx[i] > 0.0) & (ry[i] > 0.0) & (vx[i] < 0.0) & (vy[i] < 0.0) & (rz[i] > 0.0) & (vz[i] < 0.0):
		vv[i] = -vv[i]
	if (rx[i] > 0.0) & (ry[i] > 0.0) & (vx[i] < 0.0) & (vy[i] < 0.0) & (rz[i] < 0.0) & (vz[i] > 0.0):
		vv[i] = -vv[i]
	
	if (rx[i] < 0.0) & (ry[i] > 0.0) & (vx[i] > 0.0) & (vy[i] < 0.0) & (rz[i] > 0.0) & (vz[i] < 0.0):
		vv[i] = -vv[i]
	if (rx[i] < 0.0) & (ry[i] > 0.0) & (vx[i] > 0.0) & (vy[i] < 0.0) & (rz[i] < 0.0) & (vz[i] > 0.0):
		vv[i] = -vv[i]


dt = dictx['dt']

with open('vrGadget.pkl', 'rb') as f:
	vrG = pickle.load(f)
rG = vrG['r']
vr = vrG['vr']

m = dictx['m']
h = do_smoothingX((r, r))
rho = getDensity(r, m, h)

with open('rhoGadget.pkl', 'rb') as f:
	DensityG = pickle.load(f)
rhoG = DensityG['rho']


u = dictx['u'].flatten()
P = (gamma - 1.) * u * rho

with open('PGadget.pkl', 'rb') as f:
	PressureG = pickle.load(f)
PG = PressureG['P']


ax[0].cla()
ax[1].cla()
ax[2].cla()

ax[0].set_position([0.05, 0.50, 0.27, 0.40])
ax[0].scatter(rr, vv, s = 12, alpha = 1.0, color = 'k', label='This work')
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


plt.savefig('SnapShotx.png')

plt.show()







