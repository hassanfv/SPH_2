
import time
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

np.random.seed = 42

Npart = 150000 # Play with this to get the desired number of particles!

MSun = 1.98892e33

mSPH_in_g = 200.0 * MSun # mass of a single SPH particle in g.

L = 2.5/2. # half length of the cube in kpc.
V = (2.*L)**3
delta = (V/Npart)**(1./3.)

M = int(np.floor(L / delta))

pos = []

for i in range(-M, M):
    for j in range(-M, M):
        for k in range(-M, M):
            
            xt, yt, zt = 0.0+i*delta, 0.0+j*delta, 0.0+k*delta
            
            rnd = np.random.random()
            if rnd > 0.5:
                sign = 1.0
            else:
                sign = -1.0
            
            # Adding some amount of disorder
            rnd = np.random.random()
            if rnd < 1./3.:
                xt += sign * delta/4.
            if (rnd >= 1./3.) & (rnd <= 2./3.):
                yt += sign * delta/4.
            if rnd > 2./3.:
                zt += sign * delta/4.
            
            r = (xt*xt + yt*yt + zt*zt)**0.5

            pos.append([xt, yt, zt])

pos = np.array(pos)

dictx = {'r': pos, 'L': L, 'mSPH': mSPH_in_g}

with open('Grid.pkl', 'wb') as f:
	pickle.dump(dictx, f)

print(pos.shape)
plt.scatter(pos[:, 0], pos[:, 1], s = 0.1)
plt.show()




