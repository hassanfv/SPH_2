
import numpy as np
import matplotlib.pyplot as plt
import pickle

res = []
count = 0

Nparticles = 5000

while count < Nparticles:
	
	x = -1.0 + 2* np.random.random()
	y = -1.0 + 2* np.random.random()
	z = -1.0 + 2* np.random.random()

	if (x*x + y*y + z*z)**0.5 <= 1.:
		res.append([x, y, z])
		count += 1

print('Total number of particles inside the sphere = ', count)

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]


with open('Uniform_Sphere_RND.pkl', 'wb') as f:
	pickle.dump(res, f)

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = 2.);
plt.show()
