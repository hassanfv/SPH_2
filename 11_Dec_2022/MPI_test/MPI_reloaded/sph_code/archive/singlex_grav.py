

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



dirx = './Outputs/'

filez = np.sort(os.listdir(dirx))

fig, ax = plt.subplots(figsize = (7, 6))

kb = ''
res = []
tt = 0

gamma = 5./3.



j = 800

with open(dirx + filez[j], 'rb') as f:
	dictx = pickle.load(f)


r = dictx['pos']
rx = r[:, 0]
ry = r[:, 1]
rz = r[:, 2]


plt.scatter(rx, ry, s = 1, alpha = 1.0, color = 'k', label='This work')

plt.show()


plt.savefig('SnapShotx.png')








