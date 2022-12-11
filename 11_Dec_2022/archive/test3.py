
import numpy as np
import matplotlib.pyplot as plt


dx = dy = dz = 0.08

X = np.arange(-1., 1., dx)
Y = np.arange(-1., 1., dy)
Z = np.arange(-1., 1., dz)


res = []

for x in X:
	for y in Y:
		for z in Z:
			res.append([x, y, z])

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

r = np.sqrt(x * x + y * y + z * z) ** 0.5


x_new = x * r
y_new = y * r
z_new = z * r

r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

nx = np.where(r_new <= 1.0)[0]

plt.figure(figsize = (6, 6))

plt.scatter(x_new[nx], y_new[nx], s = 5)

plt.show()



