

import numpy as np
import matplotlib.pyplot as plt
import pickle
import readchar
import os
import pandas as pd
import h5py
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


kb = ''
res = []
tt = 0

gamma = 5./3.

data = h5py.File('snapshot_030.hdf5', "r")
datax = data['PartType0']
posG = datax['Coordinates']

with open('./Outputs/' + filez[300], 'rb') as f:
	dictx = pickle.load(f)


Pos = dictx['pos']
r = Pos
Velocity = dictx['v']
rho = dictx['rho'].flatten()
u = dictx['uDirectcool'].flatten()

P = (gamma - 1.) * u * rho

radius = np.sqrt(Pos[:,0]**2 + Pos[:,1]**2 + Pos[:,2]**2)
vv = (Velocity[:,0] * Pos[:,0] + Velocity[:,1] * Pos[:,1] + Velocity[:,2] * Pos[:,2]) / radius[:]
origin_particles = np.argwhere(radius == 0.0)
vv[origin_particles] = 0


bin_r = np.arange(0.01, 1.0, 0.01)
bin_v = np.zeros(len(bin_r))
bin_rho = np.zeros(len(bin_r))
bin_u = np.zeros(len(bin_r))
bin_P = np.zeros(len(bin_r))

for i in range(len(bin_r)-1):

    vt = 0.0
    rhot = 0.0
    ut = 0.0
    Pt = 0.0
    countx = 1e-10
    for j in range(len(radius)):
        if (radius[j] < bin_r[i+1]) and (radius[j] > bin_r[i]):
            
            vt += vv[j]
            rhot += rho[j]
            ut += u[j]
            Pt += P[j]
            countx += 1
    
    bin_v[i] = vt / countx
    bin_rho[i] = rhot / countx
    bin_u[i] = ut / countx
    bin_P[i] = Pt / countx


nn = np.where(bin_rho > 0.0)[0]
vv = bin_v[nn]
rho = bin_rho[nn]
u = bin_u[nn]
P = bin_P[nn]
rr = bin_r[nn]

fig, ax = plt.subplots(5, figsize = (16, 9))


with open('vrGadget.pkl', 'rb') as f:
	vrG = pickle.load(f)
rG = vrG['r']
vr = vrG['vr']



with open('rhoGadget.pkl', 'rb') as f:
	DensityG = pickle.load(f)
rhoG = DensityG['rho']


with open('PGadget.pkl', 'rb') as f:
	PressureG = pickle.load(f)
PG = PressureG['P']

ax[0].cla()
ax[1].cla()
ax[2].cla()
ax[3].cla()
ax[4].cla()


ax[0].set_position([0.05, 0.50, 0.27, 0.40])

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
ax[2].scatter(rG, vr, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[2].axis(xmin = 0.005, xmax = 1.2)
ax[2].axis(ymin = -2, ymax = 2)
ax[2].set_xlabel('Radius')
ax[2].set_ylabel('Velocity')
ax[2].set_xscale('log')
ax[2].legend()


ax[3].set_position([0.05, 0.05, 0.27, 0.40])
ax[3].scatter(rr, rho, s = 2, alpha = 1.0, color = 'k', label = 'This work')
ax[3].scatter(rG, rhoG, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[3].axis(xmin = 0.005, xmax = 1.2)
ax[3].axis(ymin = 0.01, ymax = 400)
ax[3].set_xlabel('Radius')
ax[3].set_ylabel('Density')
ax[3].set_xscale('log')
ax[3].set_yscale('log')
ax[3].legend()


ax[4].set_position([0.36, 0.05, 0.28, 0.40])
ax[4].scatter(rr, P, s = 2, alpha = 1.0, color = 'k', label = 'This work')
ax[4].scatter(rG, PG, s = 4, color = 'orange', label = 'Gadget 4.0')
ax[4].axis(xmin = 0.005, xmax = 1.2)
ax[4].axis(ymin = 0.01, ymax = 400)
ax[4].set_xlabel('Radius')
ax[4].set_ylabel('Pressure')
ax[4].set_xscale('log')
ax[4].set_yscale('log')
ax[4].legend()


plt.savefig('SnapShot.png')

plt.show()








