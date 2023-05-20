
import numpy as np
import matplotlib.pyplot as plt


def density(rho_0, R_0, R, alpha):

	return rho_0 * (R / R_0)**(-alpha)


mH = 1.6726e-24

R_0 = 100. # pc
rho_0 = 10.0 * mH # gcm^-3 # assuming fully hydrogen gas
alpha = 1.5

R = np.logspace(np.log10(0.1), np.log10(3000), 1000)

rho = density(rho_0, R_0, R, alpha)


nH = rho / mH # assuming fully hydrogen gas

plt.scatter(R, nH, s = 2, color = 'k')

plt.yscale('log')

plt.show()
