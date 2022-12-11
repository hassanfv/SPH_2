
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


plt.plot(x, y, color = 'k')
#plt.plot(x, y1, color = 'b')

plt.axhline(y = 0., linestyle = '--', color = 'b')

plt.ylim(-0.1, 1.1)


plt.show()


