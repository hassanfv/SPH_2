
# Solution of the isothermal Lane-Emden equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# See: https://scicomp.stackexchange.com/questions/2424/solving-non-linear-singular-ode-with-scipy-odeint-odepack

def dSdx(x, S):
	
	y1, y2 = S
	
	return [y2, -2./x * y2 + np.exp(-y1)]


y1_0 = 0.
y2_0 = 0.
S_0 = (y1_0, y2_0)


x = np.linspace(.00001, 10., 100)

sol = odeint(dSdx, y0 = S_0, t = x, tfirst = True)

y1_sol = sol.T[0]
y2_sol = sol.T[1]


#---- Numerical solution from https://www.worldscientific.com/doi/10.1142/S0219876207000960

yy1 = 1./6.*x**2 - 1./120.*x**4 + 1./1890.*x**6 - 61./1632960.*x**8 + 629./224532000.*x**10

yy2 = 1./3.*x - 1./30.*x**3 + 1./315.*x**5 - 61./204120.*x**7 + 629./22453200.*x**9

#--------------------------------------




plt.figure(figsize = (10, 8))

plt.plot(x, y1_sol, color = 'black', label = '\u03C8(\u03BE)') # this is psi(ksi) ........ Note: x corresponds to ksi.
plt.plot(x, y2_sol, color = 'blue',  label = 'd\u03C8/d\u03BE')  # this is d_psi/d_ksi.
plt.plot(x, x*x*y2_sol, color = 'lime', label = '$\u03BE^{2}$' + 'd\u03C8/d\u03BE')  # this is ksi^2*d_psi/d_ksi.

#plt.plot(x, yy1, color = 'yellow', label = 'y1, Guzel et al.')
#plt.plot(x, yy2, color = 'orange', label = 'y2, Guzel et al.')
#plt.plot(x, x*x*yy2, color = 'red', label = 'ksi^2 * y2, Guzel et al.')


#---------------------------------------
#x = np.arange(0.1, 10., 0.01)
#yyy1 = np.log(1.+ x**2/2. * (1. - 1.178/(2.+x**2)**0.25 * np.sin(-0.507787+7**0.5/4. * np.log(2.+x**2)) ) )

#dx = 0.01

#yyy2 = np.gradient(yyy1, dx)

#plt.plot(x, yyy1, color = 'yellow', label = 'y1, Soliman et al.')
#plt.plot(x, yyy2, color = 'orange', label = 'y2, Soliman et al.')
#plt.plot(x, x*x*yyy2, color = 'red', label = 'ksi^2 * y2, Soliman et al.')
#---------------------------------------


#---------------------------------------
x = np.arange(0.1, 10., 0.01)

A = 2./3. + 0.59858 * 1e-6 * x**6
B = (1. + 1e-6 * x**6) * (1. + x**2/15.)**0.25
C = np.cos((7.)**0.5/4. * np.log(1. + x**2/23.162231))

yyy1 = np.log(1. + x**2/2. * (1. - A/B*C))

dx = 0.01

yyy2 = np.gradient(yyy1, dx)

plt.plot(x, yyy1, color = 'yellow', label = 'y1, Soliman et al.')
plt.plot(x, yyy2, color = 'orange', label = 'y2, Soliman et al.')
plt.plot(x, x*x*yyy2, color = 'red', label = 'ksi^2 * y2, Soliman et al.')
#---------------------------------------

plt.ylim(-.1, 10)
plt.title('Isothermal Lane-Emden')

plt.legend(fontsize=15)

plt.show()


