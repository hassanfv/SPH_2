
import numpy as np
import matplotlib.pyplot as plt
import pickle



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

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]


with open('Uniform_density_sphere.pkl', 'wb') as f:
	pickle.dump(res, f)

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = .5);
plt.show()
