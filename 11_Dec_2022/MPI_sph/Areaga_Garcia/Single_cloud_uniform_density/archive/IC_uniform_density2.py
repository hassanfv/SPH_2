
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)


res = []
count = 0

N = 10000

while count < N:

	x = 1. - 2. * np.random.random()
	y = 1. - 2. * np.random.random()
	z = 1. - 2. * np.random.random()

	if (x**2 + y**2 + z**2)**0.5 <= 1.:
		res.append([x, y, z])
		count += 1

print('Total number of particles inside the sphere = ', count)

r = np.array(res)

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = .1);
plt.show()

rr = (x**2 + y**2 + z**2)**0.5

m = 1. / N # the mass of each particle

res = []

rho_0 = 0.1

for i in range(N):

	# pick a particle from the uniform sphere and calculate its r, theta, and phi:
	xt = r[i, 0]
	yt = r[i, 1]
	zt = r[i, 2]

	r_unif_t = (xt*xt + yt*yt + zt*zt)**0.5
	
	# calculating the corresponding theta & phi
	theta = np.arccos(zt/r_unif_t) # the angle from z-axis
	phi = np.arctan(yt/xt)
	
	if (xt < 0.) & (yt > 0.):
		phi += np.pi
	
	elif (xt < 0.) & (yt < 0.):
		phi += np.pi
	
	elif (xt > 0.) & (yt < 0.):
		phi += 2.*np.pi
		

	#--- calculating Mass in initial particle distribution
	nnr = np.where(rr <= r_unif_t)[0]
	Mass_in_r_unif_t = len(nnr) * m

	# now we find the best index for which the summation gets almost equal to Mass_in_r_unif_t	
	r_dist = 3.*Mass_in_r_unif_t/4./np.pi/rho_0
	
	x_dist = r_dist * np.cos(phi) * np.sin(theta)
	y_dist = r_dist * np.sin(phi) * np.sin(theta)
	z_dist = r_dist * np.cos(theta)

	res.append([x_dist, y_dist, z_dist])

res = np.array(res)

with open('Uniform_density_sphere.pkl', 'wb') as f:
	pickle.dump(res, f)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = .1);
plt.show()





