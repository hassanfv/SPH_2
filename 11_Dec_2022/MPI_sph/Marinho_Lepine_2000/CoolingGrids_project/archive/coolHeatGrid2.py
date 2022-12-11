
import numpy as np
import matplotlib.pyplot as plt
import time
from coolHeat2 import *


unit_U = 9537129466.43166
UnitTime_in_s = 3.16e13 # == 1Myr
unit_rho = 1.5e-20 # g/cm3


u_arr = np.logspace(np.log10(0.1), np.log10(13400.), 200)

def LambdaGrid(rho):

	rho_cgs = rho * unit_rho

	res = []

	for i in range(len(u_arr)):

		#---- calculating some rough T_init ----
		y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
		mu_approx = X/2. + X*y/2. + Y/4.
		T_init = (gamma - 1.0) * mH / kB * mu_approx * u_arr[i] * unit_U
		#---------------------------------------

		res.append([u_arr[i], -coolingRateFromU_M(u_arr[i] * unit_U, T_init, rho_cgs)])


	return np.array(res)


rho2 = 0.01
Lambdax2 = LambdaGrid(rho2)
uu0 = Lambdax2[:, 0]
Lambda0 = Lambdax2[:, 1]

rho1 = .10
Lambdax1 = LambdaGrid(rho1)
uu1 = Lambdax1[:, 0]
Lambda1 = Lambdax1[:, 1]

rho2 = 1.0
Lambdax2 = LambdaGrid(rho2)
uu2 = Lambdax2[:, 0]
Lambda2 = Lambdax2[:, 1]

rho2 = 10.0
Lambdax2 = LambdaGrid(rho2)
uu3 = Lambdax2[:, 0]
Lambda3 = Lambdax2[:, 1]


plt.plot(uu0, Lambda0, color = 'black')
plt.plot(uu1, Lambda1, color = 'orange')
plt.plot(uu2, Lambda2, color = 'blue')
plt.plot(uu3, Lambda3, color = 'lime')

plt.xscale('log')
plt.yscale('log')

plt.show()






