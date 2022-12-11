
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(42)


nPart = 40000

res = []

M = 1.
rho0 = 3./4./np.pi

for i in range(nPart):

	ksi1, ksi2, ksi3 = np.random.random(3)

	theta = np.arccos(1. - 2. * ksi2)
	phi = 2. * np.pi * ksi3
	
	r = (3.*M*ksi1/(4.*np.pi*rho0*(1.+0.1*np.cos(2.*phi))))**(1./3.)
	
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	
	res.append([x, y, z])


res = np.array(res)

plt.figure(figsize = (8, 8))

plt.scatter(res[:, 0], res[:, 1], s = 0.05, color = 'black')

#plt.xlim(-24, 24)
#plt.ylim(-24, 24)

plt.show()	







