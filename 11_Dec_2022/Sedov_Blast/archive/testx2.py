
import numpy as np
from do_smoothingZ import *
import time
np.random.seed(10)


pos = np.random.normal(0, 1, (12000, 3))

x = pos[:, 0]
y = pos[:, 1]
z = pos[:, 2]


TA = time.time()

N = pos.shape[0]

nSplit = 6

M = int(np.floor(N/nSplit))

hres = []

for k in range(nSplit-1):

	posT = pos[k:(k+M), :]

	xt = posT[:, 0].reshape((M, 1))
	yt = posT[:, 1].reshape((M, 1))
	zt = posT[:, 2].reshape((M, 1))

	dx = xt - x.T
	dy = yt - y.T
	dz = zt - z.T

	dx2 = dx*dx
	dy2 = dy*dy
	dz2 = dz*dz

	dist = (dx2 + dy2 + dz2)**0.5

	print('shape = ', dist.shape)
	
	#dist.sort(axis = 1)
	
	xxx = dist < 2.0
	
	#hres += list(dist[:, 64])
	


#print(np.array(hres))

print(xxx)	
	
	
	

print('Elapsed time = ', time.time() - TA)







