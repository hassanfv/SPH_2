
import numpy as np
import matplotlib.pyplot as plt


dx = dy = 0.1

X = np.arange(-1.25, 1.25, dx)
Y = np.arange(-1.25, 1.25, dy)


res = []

for x in X:
	for y in Y:
		res.append([x, y])

res = np.array(res)

x = res[:, 0]
y = res[:, 1]

r = np.sqrt(x * x + y * y)


sinT = r / y
cosT = r / x

r_new = r ** (3./2.)
x_new = r_new / cosT
y_new = r_new / sinT


plt.figure(figsize = (6, 6))

plt.scatter(x_new, y_new, s = 5)

plt.show()



