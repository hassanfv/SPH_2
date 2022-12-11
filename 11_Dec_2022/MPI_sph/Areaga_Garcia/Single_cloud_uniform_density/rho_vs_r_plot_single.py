
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
import readchar
from libsx import *


TA = time.time()

with open('Uniform_density_sphere.pkl', 'rb') as f:
	r = pickle.load(f)


x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

rr = (x*x + y*y + z*z)**0.5

print('rr = ', np.sort(rr))


N = r.shape[0]
m = 1.0 / N + np.zeros(N)

h = do_smoothingX((r, r))

print('h = ', np.sort(h))

rho = getDensity(r, m, h)

print('rho = ', np.sort(rho))

print('Elapsed time = ', time.time() - TA)

plt.hist(rho, bins = np.arange(0, 1, 0.05))
plt.show()

plt.scatter(rr, rho, s = 0.4, color = 'k')

plt.savefig('fig.png')

plt.show()


	
	




