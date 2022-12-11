
import numpy as np
import matplotlib.pyplot as plt


dx = dy = dz = 0.05

X = np.arange(-.5, .5, dx)
Y = np.arange(-.5, .5, dy)
Z = np.arange(-.5, .5, dz)


res = []

for x in X:
	for y in Y:
		for z in Z:
			res.append([x, y, z])

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

r = np.sqrt(x * x + y * y + z * z)

cosTheta = z / r
Theta = np.arccos(cosTheta)
sinTheta = np.sin(Theta)

sinPHI = x / (r * sinTheta)
cosPHI = y / (r * sinTheta)

r_new = r ** (3./2.)

x_new = r_new * sinTheta * sinPHI
y_new = r_new * sinTheta * cosPHI
z_new = r_new * cosTheta


plt.figure(figsize = (6, 6))

plt.scatter(x_new, y_new, s = 5)

plt.show()



