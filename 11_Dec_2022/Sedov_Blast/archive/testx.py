
import numpy as np
from do_smoothingZ import *
import time


pos = np.random.normal(0, 1, (10000, 3))


print(pos.shape)

N = pos.shape[0]

TA = time.time()

x = pos[:, 0].reshape((N, 1))
y = pos[:, 1].reshape((N, 1))
z = pos[:, 2].reshape((N, 1))

dx = x - x.T
dy = y - y.T
dz = z - z.T

dx2 = dx*dx
dy2 = dy*dy
dz2 = dz*dz

dist = (dx2 + dy2 + dz2)**0.5

print(dist.shape)

print('Elapsed time = ', time.time() - TA)







