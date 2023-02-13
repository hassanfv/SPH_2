
# ref: Zubovas & Bourne (2017)

import numpy as np
import matplotlib.pyplot as plt

#===== rho_func
def rho_func(R_in_pc, C):
	
	R_cm = R_in_pc * 3.086e18
	
	return C/R_cm**2




M_sun = 1.98892e33 # g
M_sh = 6.1e9 * M_sun

R_in = 0.1 # kpc
R_out = 2.0 # kpc

R_in_cm = R_in * 1000.0 * 3.086e18
R_out_cm = R_out * 1000.0 * 3.086e18

C = M_sh / 4. / np.pi / (R_out_cm - R_in_cm)

Rgrid = np.linspace(100, 2000, 100)

rho = [rho_func(r, C) for r in Rgrid]

XH = 0.76
mH = 1.6726e-24

nH = [rhot*XH/mH for rhot in rho]

plt.scatter(Rgrid, nH, s = 1)

plt.yscale('log')
plt.show()





