
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-1., 1., 0.1)
y = np.arange(-1., 1., 0.1)
z = np.arange(-1., 1., 0.1)

res = []

for i in range(len(x)):

	for j in range(len(x)):
	
		for k in range(len(x)):
		
			res.append([x[i], y[j], z[k]])


res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = 2.);
plt.show()
