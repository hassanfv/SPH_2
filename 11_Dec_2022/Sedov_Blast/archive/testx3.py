
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

        hres.append(np.sort(dist)[45])

    return np.array(hres) * 0.5





#===== smooth_hX (non-parallel)
@njit
def do_smoothingXX(pos, h):

	N = pos.shape[0]
	
	N_up = 60
	N_low = 40

	hres = []
	

	for i in range(N):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5


		Nt = np.sum(dist < 2.0*hi)
		#print('A = ', Nt, hi)
			
		while (Nt < N_low) or (Nt > N_up):
			
			if Nt < N_low:
				hi += 0.02 * hi

			elif Nt > N_up:
				hi -= 0.02 * hi
			
			Nt = np.sum(dist < 2.0*hi)
			
			#print(Nt, hi)
			
			#kb = readchar.readkey()
			#if kb == 'q':
			#	break
			

		hres.append(hi)
		

	return hres



with open('SedovBlast.pkl', 'rb') as f:
    res = pickle.load(f)
resx = res[:, 0].reshape((len(res[:, 0]),1))
resy = res[:, 1].reshape((len(res[:, 1]),1))
resz = res[:, 2].reshape((len(res[:, 2]),1))

pos = np.hstack((resx, resy, resz))

print(pos.shape)

TA = time.time()
h1 = do_smoothingX((pos, pos))
print('Elapsed time (A) = ', time.time() - TA)



TB = time.time()

ht = 0.9 * h1

h = do_smoothingXX(pos, ht)

print(np.sort(h1))
print(np.sort(h))

#for i in range(len(h1)):
#	print(h1[i], h[i])
	

print('Elapsed time (B) = ', time.time() - TB)







