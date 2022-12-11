
import numpy as np
import matplotlib.pyplot as plt


T = 1000.
X = 0.75
Y = 0.25
mH = 1.6726e-24 # gram

n_tot = 100

rho = mH * n_tot / (X + Y/4.)

print('rho = ', rho)

Tarr = np.arange(100, 10000, 100)

res = []

for T in Tarr:

	K = 2.11 / rho / X * np.exp(-52490./T)

	delta = K*K + 4. * K

	y = (-K + delta**0.5) / 2.

	nHI = rho * X / mH * y
	nH2 = rho * X / (2.*mH) * (1. - y)

	res.append([T, nHI, nH2, y])


res = np.array(res)

nHI = res[:, 1]
nH2 = res[:, 2]

y = res[:, 3]


#plt.plot(Tarr, nHI, color = 'black')
#plt.plot(Tarr, nH2, color = 'blue')
plt.plot(Tarr, y, color = 'blue')
#plt.xlim(500, 4500)

plt.show()




