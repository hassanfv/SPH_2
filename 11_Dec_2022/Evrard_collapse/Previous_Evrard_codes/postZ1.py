

import numpy as np
import matplotlib.pyplot as plt
import pickle
import readchar
import os
import pandas as pd
import h5py
import time
#from cool_libs import *

FloatType = np.float64


def getPairwiseSeparations_h(r, pos):
    
    M = r.shape[0]
    N = pos.shape[0]
    
    rx = r[:, 0].reshape((M, 1))
    ry = r[:, 1].reshape((M, 1))
    rz = r[:, 2].reshape((M, 1))
    
    posx = pos[:, 0].reshape((N, 1))
    posy = pos[:, 1].reshape((N, 1))
    posz = pos[:, 2].reshape((N, 1))
    
    dx = rx - posx.T
    dy = ry - posy.T
    dz = rz - posz.T
    
    return dx, dy, dz
    

def W_I(dx, dy, dz, h):

    rr = np.sqrt(dx**2 + dy**2 + dz**2)

    sig = 1.0/np.pi
    q = rr / h

    Wq1 = 1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3
    ntt = np.where(q > 1.0)
    rows = ntt[0]
    cols = ntt[1]
    Wq1[rows, cols] = 0.0


    Wq2 = (1.0/4.0) * (2.0 - q)**3
    ntt = np.where(np.logical_or(q <= 1.0, q > 2.0))
    rows = ntt[0]
    cols = ntt[1]
    Wq2[rows, cols] = 0.0

    return  sig / h**3 * (Wq1 + Wq2)


def getDensity(r, pos, m, h):  # You may want to change your Kernel !!

    dx, dy, dz = getPairwiseSeparations_h(r, pos)
    M = r.shape[0]
    rho = np.sum(m * W_I(dx, dy, dz, h), 1).reshape((M, 1))
    
    return rho



def smooth_h(pos):

    N = pos.shape[0]
    dx, dy, dz = getPairwiseSeparations_h(pos, pos)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    hres = []

    for j in range(N):

        rrt = dist[j, :]
        rrx = np.sort(rrt)
        hres.append(rrx[64])

    return np.array(hres).reshape((1, N))


filez = np.sort(os.listdir('./Outputs/'))

fig, ax = plt.subplots(5, figsize = (16, 9))

kb = ''
res = []
tt = 0

gamma = 5./3.

data = h5py.File('snapshot_008.hdf5', "r")
datax = data['PartType0']
posG = datax['Coordinates']



def get_data_bins(radius,Density, vr, A):
	number_bins = 100
	min_r = 0.01
	max_r = 1.0

	r_bin = np.zeros(number_bins)
	vr_bin = np.zeros(number_bins)

	rho_bin = np.zeros(number_bins)
	count_bin = np.zeros(number_bins) + 1
	entropy_bin = np.zeros(number_bins)

	for i in range(radius.size):
		if(radius[i] < min_r or radius[i] > max_r):
			continue
		bin_value = int((np.log10(radius[i] / min_r))/(np.log10(max_r/min_r)) * number_bins)
		count_bin[bin_value] = count_bin[bin_value] + 1
		vr_bin[bin_value] = vr_bin[bin_value] + vr[i]
		rho_bin[bin_value] = rho_bin[bin_value] + Density[i]
		entropy_bin[bin_value] = entropy_bin[bin_value] + A[i]


	vr_bin /= count_bin
	rho_bin /= count_bin
	entropy_bin /= count_bin

	for i in range(number_bins):
		r_bin[i] = (i+0.5)* (np.log10(max_r/min_r)/number_bins) + np.log10(min_r)
		r_bin[i] = 10**r_bin[i]
		print(count_bin[i])

	return r_bin,rho_bin, vr_bin, entropy_bin




with open('./Outputs/' + filez[80], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
Pos = r
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]

rr = np.sqrt(rx**2 + ry**2 + rz**2)
	
v = dictx['v']
Velocity = v
vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

vv = np.sqrt(vx**2 + vy**2 + vz**2)

tarr = np.zeros(r.shape[0])


radius = np.sqrt(Pos[:,0]**2 + Pos[:,1]**2 + Pos[:,2]**2)
vr = (Velocity[:,0] * Pos[:,0] + Velocity[:,1] * Pos[:,1] + Velocity[:,2] * Pos[:,2]) / radius[:]
origin_particles = np.argwhere(radius == 0.0)
vr[origin_particles] = 0


dt = dictx['dt']

with open('vrGadget.pkl', 'rb') as f:
	vrG = pickle.load(f)
rG = vrG['r']
#vr = vrG['vr']

mSPH = dictx['m']
#h = smooth_h(r)
#hij = (h.T + h)/2.0
rho = dictx['rho']  #getDensity(r, r, mSPH, hij).flatten()

with open('rhoGadget.pkl', 'rb') as f:
	DensityG = pickle.load(f)
rhoG = DensityG['rho']


u = dictx['uDirectcool'].flatten()

with open('PGadget.pkl', 'rb') as f:
	PressureG = pickle.load(f)
PG = PressureG['P']


Density = rho
A = u

rr, vv, rho, u = get_data_bins(radius,Density, vr, A)

print(vv)


P = (gamma - 1.) * u * rho

tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


ax[0].cla()
ax[1].cla()
ax[2].cla()
ax[3].cla()
ax[4].cla()


ax[0].set_position([0.05, 0.50, 0.27, 0.40])

N = r.shape[0]
ax[0].scatter(posG[:, 0], posG[:, 1], s = 1, alpha = 1.0, color = 'orange', label = 'Gadget 4.0')
ax[0].scatter(r[:, 0], r[:, 1], s = 0.5, alpha = 1.0, color = 'black', label = 'This work')
rang = 1.10
ax[0].axis(xmin = -rang, xmax = rang)
ax[0].axis(ymin = -rang, ymax = rang)
ax[0].text(-0.9, 0.85, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend(loc = 'upper right')

ax[1].set_position([0.36, 0.50, 0.28, 0.40])
ax[1].scatter(posG[:, 0], posG[:, 2], s = 1, alpha = 1.0, color = 'orange', label = 'Gadget 4.0')
ax[1].scatter(r[:, 0], r[:, 2], s = 0.5, alpha = 1.0, color = 'black', label = 'This work')
ax[1].axis(xmin = -rang, xmax = rang)
ax[1].axis(ymin = -rang, ymax = rang)
ax[1].text(-0.9, 0.85, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[1].set_xlabel('X')
ax[1].set_ylabel('Z')
ax[1].legend(loc = 'upper right')

ax[2].set_position([0.68, 0.50, 0.28, 0.40])
ax[2].scatter(rr, vv, s = 2, alpha = 1.0, color = 'black', label = 'This work')
#ax[2].scatter(rG, vr, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[2].axis(xmin = 0.005, xmax = 1.2)
ax[2].axis(ymin = -2, ymax = 2)
ax[2].set_xlabel('Radius')
ax[2].set_ylabel('Velocity')
ax[2].set_xscale('log')
ax[2].legend()


ax[3].set_position([0.05, 0.05, 0.27, 0.40])
ax[3].scatter(rr, rho, s = 2, alpha = 1.0, color = 'k', label = 'This work')
#ax[3].scatter(rG, rhoG, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[3].axis(xmin = 0.005, xmax = 1.2)
ax[3].axis(ymin = 0.01, ymax = 400)
ax[3].set_xlabel('Radius')
ax[3].set_ylabel('Density')
ax[3].set_xscale('log')
ax[3].set_yscale('log')
ax[3].legend()


ax[4].set_position([0.36, 0.05, 0.28, 0.40])
ax[4].scatter(rr, P, s = 2, alpha = 1.0, color = 'k', label = 'This work')
#ax[4].scatter(rG, PG, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[4].axis(xmin = 0.005, xmax = 1.2)
ax[4].axis(ymin = 0.01, ymax = 400)
ax[4].set_xlabel('Radius')
ax[4].set_ylabel('Pressure')
ax[4].set_xscale('log')
ax[4].set_yscale('log')
ax[4].legend()


#fig.canvas.flush_events()

#time.sleep(0.01)

plt.show()

plt.savefig('SnapShot.png')








