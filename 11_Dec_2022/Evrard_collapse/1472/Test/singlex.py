

import numpy as np
import matplotlib.pyplot as plt
import pickle
import readchar
import os
import pandas as pd
import time
#from cool_libs import *


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
        hres.append(rrx[32])

    return np.array(hres).reshape((1, N))


filez = np.sort(os.listdir('./Outputs/'))

fig, ax = plt.subplots(3, figsize = (16, 9))

kb = ''
res = []
tt = 0

gamma = 5./3.



j = 800

with open('./Outputs/' + filez[j], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]

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

mSPH = dictx['m']
h = smooth_h(r)
hij = (h.T + h)/2.0
rho = getDensity(r, r, mSPH, hij).flatten()

with open('rhoGadget.pkl', 'rb') as f:
	DensityG = pickle.load(f)
rhoG = DensityG['rho']


u = dictx['uDirectcool'].flatten()
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
ax[0].text(0.01, 1.4, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[0].axis(xmin = 0.005, xmax = 1.2)
ax[0].axis(ymin = -2, ymax = 2)
ax[0].set_xlabel('Radius')
ax[0].set_ylabel('Velocity')
ax[0].set_xscale('log')
ax[0].legend()


ax[1].set_position([0.36, 0.50, 0.28, 0.40])
ax[1].scatter(rr, rho, s = 12, alpha = 1.0, color = 'k', label='This work')
ax[1].scatter(rG, rhoG, s = 10, color = 'lime', label = 'Gadget 4')
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
ax[2].text(0.01, 150, 't = ' + str(np.round(tt, 3)), fontsize = 16)
ax[2].axis(xmin = 0.005, xmax = 1.2)
ax[2].axis(ymin = 0.01, ymax = 400)
ax[2].set_xlabel('Radius')
ax[2].set_ylabel('Pressure')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].legend()


#plt.show()


plt.savefig('SnapShotx.png')








