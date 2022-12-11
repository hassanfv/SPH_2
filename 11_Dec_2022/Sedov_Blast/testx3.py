
import numpy as np
import time
from numba import jit, njit
import pickle
import readchar
import matplotlib.pyplot as plt
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

	Nth_up = 50 + 10.
	Nth_low = 50 - 10.

	hres = np.zeros_like(h)

	for i in range(N):

		hi = h[i]
		dist = np.zeros(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5


		Nngb = np.sum(dist < 2.0*hi)
		#print('A = ', Nngb, hi)

		if i == 762: print('(X): i, Nngb, Nth_low, Nth_up, h[i] = ', i, Nngb, Nth_low, Nth_up, np.round(h[i], 5))

		kk = 0

		while ((Nngb > Nth_up) or (Nngb < Nth_low)) and (i == 762):
		
			print('(1): i, Nngb, Nth_low, Nth_up, hi, h[i] = ', i, Nngb, Nth_low, Nth_up, np.round(hi, 5), np.round(h[i], 5))

			if Nngb > Nth_up:

				#pass
				#print('1) hi before = ', hi)
				hi -= 0.05 * hi
				#print('1) hi After = ', hi)

			#elif Nngb < Nth_low:
			#else:
			#if Nngb < Nth_low:
				
				#pass
				#print('2) hi before = ', hi)
			#	hi += 0.00001 * hi
				#print('2) hi After = ', hi)

			print('(0): i, Nngb, Nth_low, Nth_up, hi, h[i] = ', i, Nngb, Nth_low, Nth_up, np.round(hi, 5), np.round(h[i], 5))
			Nngb = np.sum(dist < 2.0*hi)

			print('(2): i, Nngb, Nth_low, Nth_up, hi, h[i] = ', i, Nngb, Nth_low, Nth_up, np.round(hi, 5), np.round(h[i], 5))
			print()
			#print(i, j, np.round(hi, 5), np.round(h[i], 5), Nngb, Nth_low, Nth_up)

			kk += 1

			if kk == 100:
				break


		hres[i] = hi

		#if i == 10:
		#	break

	return hres



with open('SedovBlast.pkl', 'rb') as f:
    res = pickle.load(f)
resx = res[:, 0].reshape((len(res[:, 0]),1))
resy = res[:, 1].reshape((len(res[:, 1]),1))
resz = res[:, 2].reshape((len(res[:, 2]),1))


plt.scatter(resx, resy, s = 2, color = 'k')
plt.scatter(resx[762, 0], resy[762, 0], s = 100, color = 'red')

plt.show()

s()


pos = np.hstack((resx, resy, resz))

print(pos.shape)

TA = time.time()
h1 = do_smoothingX((pos, pos))
print('Elapsed time (A) = ', time.time() - TA)


TB = time.time()

ht = h1 + 0.1 * h1

h = do_smoothingXX(pos, ht)

print((h1))
print((h))

#for i in range(len(h1)):
#	print(h1[i], h[i])
	

print('Elapsed time (B) = ', time.time() - TB)







