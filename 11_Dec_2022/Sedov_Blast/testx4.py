
import numpy as np
import time
from numba import jit, njit
import pickle
import readchar
np.random.seed(10)


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
            dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

        hres.append(np.sort(dist)[40])

    return np.array(hres) * 0.5





#===== smooth_hX (non-parallel)
@njit
def do_smoothingXX(pos, h):

	N = pos.shape[0]

	Nth_up = 40 + 5.
	Nth_low = 40 - 5.

	hres = np.zeros_like(h)
	
	n_Max_iteration = 100

	for i in range(N):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5


		Nngb = np.sum(dist < 2.0*hi)

		#n_iter = 0

		while (Nngb > Nth_up) or (Nngb < Nth_low):
		
			if Nngb > Nth_up:

				hi -= 0.003 * hi

			if Nngb < Nth_low:
				
				hi += 0.003 * hi

			Nngb = np.sum(dist < 2.0*hi)
			
			#n_iter += 1

		hres[i] = hi

	return hres



#with open('SedovBlast.pkl', 'rb') as f:
#    res = pickle.load(f)
#resx = res[:, 0].reshape((len(res[:, 0]),1))
#resy = res[:, 1].reshape((len(res[:, 1]),1))
#resz = res[:, 2].reshape((len(res[:, 2]),1))

with open('Evrard_5616.pkl', 'rb') as f:
    res = pickle.load(f)

resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))

pos = np.hstack((resx, resy, resz))

print(pos.shape)

TA = time.time()
h1 = do_smoothingX((pos, pos))
print('Elapsed time (A) = ', time.time() - TA)


TB = time.time()

ht = h1 + .1 * h1

h = do_smoothingXX(pos, ht)

print((h1))
print((h))

#for i in range(len(h1)):
#	print(h1[i], h[i])
	

print('Elapsed time (B) = ', time.time() - TB)







