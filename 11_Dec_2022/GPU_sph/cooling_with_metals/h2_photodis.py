import numpy as np
import matplotlib.pyplot as plt
from photolibs3 import *


#------- sig_L0_func --------------
def sig_L0_func(h_nu):

	if (h_nu >= 14.675) & (h_nu < 16.820):

		sig_L0 = 1e-18 * 10**(15.128 - 1.05139 * h_nu)
		return sig_L0

	if (h_nu >= 16.820) & (h_nu < 17.6):

		sig_L0 = 1e-18 * 10**(-31.41 + 1.8042 * 1e-2 * h_nu**3 - 4.2339 * 1e-5 * h_nu**5)
		return sig_L0
	
	return 0.0
	

#------- sig_W0_func ---------------
def sig_W0_func(h_nu):

	if (h_nu >= 14.675) & (h_nu <= 17.7):
		
		sig_W0 = 1e-18 * 10**(13.5311 - 0.9182618 * h_nu)
		return sig_W0
		
	return 0.0


#------- sig_L1_func -----------------
def sig_L1_func(h_nu):

	if (h_nu >= 14.159) & (h_nu <= 15.302):
		
		sig_L1 = 1e-18 * 10**(12.0218406 - 0.819429 * h_nu)
		return sig_L1
		
	if (h_nu >= 15.302) & (h_nu < 17.2):

		sig_L1 = 1e-18 * 10**(16.04644 - 1.082438 * h_nu)
		return sig_L1
		
	return 0.0


#------- sig_W1_func ---------------
def sig_W1_func(h_nu):

	if (h_nu >= 14.159) & (h_nu <= 17.2):

		sig_W1 = 1e-18 * 10**(12.87367 - 0.85088597 * h_nu)
		return sig_W1
		
	return 0.0


#------- A typical Eclipsing DLA Quasar -------
L_912 = 3.25e42 # erg/s/A
L_912 = flamb_to_fneu(L_912, 912.0) # L is now in erg/s/Hz
dist = 500.0 # pc
dist_cm = pc_to_cm(dist)
J912QSO = 1.0/(4. * np.pi) * L_912 / (4. * np.pi * dist_cm**2) # J at neu = 912 A. We assume a flat spectrum from 1 to 20 Ryd (Fathivavsari et al. 2015).
#----------------------------------------------


#sig = 1.0/(y + 1.0) * (sig_L0 + sig_W0) + (1.0 - 1.0 / (y + 1.0)) * (sig_L1 + sig_W1)



h_nu = 17.6 # eV
print(eV_to_ryd(h_nu))

h_nu = np.arange(14, 17.5, 0.1)

sigL0 = [sig_L0_func(x) for x in h_nu]
sigL1 = [sig_L1_func(x) for x in h_nu]

sigW0 = [sig_W0_func(x) for x in h_nu]
sigW1 = [sig_W1_func(x) for x in h_nu]

plt.scatter(h_nu, sigL0, s = 10, color = 'black', label = 'sigL0')
plt.scatter(h_nu, sigL1, s = 10, color = 'blue', label = 'sigL1')

plt.scatter(h_nu, sigW0, s = 10, color = 'lime', label = 'sigW0')
plt.scatter(h_nu, sigW1, s = 10, color = 'cyan', label = 'sigW1')

plt.legend()

plt.show()







