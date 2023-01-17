import numpy as np
import matplotlib.pyplot as plt
from photolibs3 import *


#===== ryd_to_neu
def ryd_to_neu(ryd):

	neu = eV_to_neu(ryd_to_eV(ryd))
	
	return neu


#===== photoCrossSection 
def photoCrossSection(neu, E0, sig0, ya, P, yw, y0, y1): # Calculates cross section for each frequency!!

	E = neu_to_eV(neu)
		
	x = E/E0 - y0
	y = np.sqrt(x**2 + y1**2)
	Fy = (((x - 1)**2 + yw**2) * y**(0.5*P - 5.5)) * (1.0 + np.sqrt(y/ya))**(-P)
	
	sig = sig0 * Fy * 1e-18
	
	return sig


#===== s_nu_func
def s_nu_func(x):

	coeff = 1.0 / 5.5

	if x < 1.0:
		return coeff * 5.5
	
	if (x >= 1.0) & (x < 2.5):
		return coeff * x ** (-1.8)
	
	if (x >= 2.5) & (x < 4.0):
		return coeff * 0.4 * x**(-1.8)
	
	if x >= 4.0:
		return coeff * 2e-3 * x**3 / (np.exp(x/1.4) - 1)


#===== tau_neu_func
def tau_neu_func(neu, tau_0):

	sigHI_0 = photoCrossSection(eV_to_neu(13.60), 4.298E-1, 5.475E4, 3.288E1, 2.963, 0.0, 0.0, 0.0)
	sigHI_neu = photoCrossSection(neu, 4.298E-1, 5.475E4, 3.288E1, 2.963, 0.0, 0.0, 0.0)
	sigHeI_neu = photoCrossSection(neu, 1.361E+1, 9.492E2, 1.469E0, 3.188, 2.039, 4.434E-1, 2.136)

	tau_neu = tau_0 / sigHI_0 * (0.76 * sigHI_neu + 0.06 * sigHeI_neu)
	
	return tau_neu



#===== J_nu_func
def J_nu_func(xt, tau_0, alpha, fQ, J_0): # xt is in rydberg

	s_nu = s_nu_func(xt)
	tau = tau_neu_func(ryd_to_neu(xt), tau_0)
	J_nu = J_0 * np.exp(-tau) * (1.0 / (1.0 + fQ) * s_nu + fQ / (1.0 + fQ) * xt**(-alpha))
	
	return J_nu



alpha = 2.0
fQ = 10
tau_0 = 0.01
#J_0 = 1.0

x = ryd = np.logspace(np.log10(0.1), np.log10(60), 1000)

J_nu = np.zeros_like(x)


#------- A typical Eclipsing DLA Quasar -------
L_912 = 3.25e42 # erg/s/A
L_912 = flamb_to_fneu(L_912, 912.0) # L is now in erg/s/Hz
dist = 1e6 #500.0 # pc
dist_cm = pc_to_cm(dist)
J912QSO = 1.0/(4. * np.pi) * L_912 / (4. * np.pi * dist_cm**2) # J at neu = 912 A or 1.0 ryd.
print(f'J912QSO = {J912QSO:.3E}')
print(f'log J912QSO = {np.log10(J912QSO):.3f}')
#----------------------------------------------

J_0 = J912QSO

x = np.logspace(np.log10(0.1), np.log10(60), 100) # in ryd

J_nu = [J_nu_func(xt, tau_0, alpha, fQ, J_0) for xt in x]


plt.scatter(x, np.log10(J_nu), s = 10)

plt.xlim(0.4, 60)
#plt.ylim(-6.5, 2)

plt.xscale('log')

plt.show()






