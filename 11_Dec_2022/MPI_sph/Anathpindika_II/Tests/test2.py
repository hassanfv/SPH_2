
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def dSdx(x, S):

	y, y1 = S
	
	return [y1, -2./x*y1 - y**3]


y_0 = 1.
y1_0 = 0.

S_0 = (y_0, y1_0)

x = np.linspace(0.01, 10., 1000)
x = np.arange(0.01, 10., 0.01)

sol = odeint(dSdx, y0 = S_0, t = x, tfirst = True)

y = sol.T[0]
y1= sol.T[1]


i = 1
M = []

s = 0.

while x[i] < 6.9:

	s += (x[i] - x[i-1]) * ( (x[i-1]**2 * y[i-1]**3) + x[i]**2 * y[i]**3 ) / 2.0  # Trapezoidal rule.
	
	M.append([x[i-1], s])
	
	i += 1


M = np.array(M)

x = M[:, 0]
Mass = M[:, 1]



plt.plot(x/max(x), Mass/max(Mass), color = 'k')


plt.show()


