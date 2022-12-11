
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


T_arr = np.logspace(1, 6, 100)

Lambda = []

for Temp in T_arr:

	nHI, nH2, n_tot = h1_h2_abundance(Temp, rho_cgs)
	
	utt = u_as_a_func_of_T(Temp, rho_cgs)
	
	Lambda.append([Temp, cooling_rate(Temp, nHI, nH2, n_tot), utt/unit_U])


Lambda = np.array(Lambda)

TT = Lambda[:, 0]
uu = Lambda[:, 2]
Lambda = Lambda[:, 1]

plt.plot(uu, Lambda)

plt.xscale('log')
plt.yscale('log')

plt.show()



