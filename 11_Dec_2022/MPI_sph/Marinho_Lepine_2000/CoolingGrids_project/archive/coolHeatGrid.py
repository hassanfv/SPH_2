
import numpy as np
import matplotlib.pyplot as plt
import time
from coolHeat2 import *


unit_U = 9537129466.43166
UnitTime_in_s = 3.16e13 # == 1Myr
unit_rho = 1.5e-20 # g/cm3



rho = .10
u = 2.
dt = 0.01

rho_cgs = rho * unit_rho

u_arr = np.logspace(np.log10(0.1), np.log10(13400.), 500)

res = []

for i in range(len(u_arr)):

	#---- calculating some rough T_init ----
	y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
	mu_approx = X/2. + X*y/2. + Y/4.
	T_init = (gamma - 1.0) * mH / kB * mu_approx * u_arr[i] * unit_U
	#---------------------------------------

	res.append([u_arr[i], -coolingRateFromU_M(u_arr[i] * unit_U, T_init, rho_cgs)])


Lambdax = np.array(res)

uu = Lambdax[:, 0]
Lambda = Lambdax[:, 1]


plt.plot(uu, Lambda)

plt.xscale('log')
plt.yscale('log')

plt.show()






