
import numpy as np
import matplotlib.pyplot as plt
import pickle


stp = 0.10
x = np.arange(-1.1, 1.1, stp)
y = np.arange(-1.1, 1.1, stp)
z = np.arange(-1.1, 1.1, stp)

res = []
count = 0

for i in range(len(x)):

	for j in range(len(x)):
	
		for k in range(len(x)):
		
			if (x[i]*x[i] + y[j]*y[j] + z[k]*z[k])**0.5 <= 1.:
				res.append([x[i], y[j], z[k]])
				count += 1

print('Total number of particles inside the sphere = ', count)

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]


with open('Uniform_Sphere.pkl', 'wb') as f:
	pickle.dump(res, f)

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = 2.);
plt.show()
