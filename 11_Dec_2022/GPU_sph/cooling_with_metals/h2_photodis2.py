import numpy as np
import matplotlib.pyplot as plt
from photolibs3 import *


#===== eV_to_erg
def eV_to_erg(eV):

	return eV * 1.60218e-12


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


#------- sig_H2_photo_dissociation ---------------
def sig_H2_photo_dissociation(h_nu):

	sig = np.zeros_like(h_nu)
	
	y = 3.0

	for i in range(len(h_nu)):

		sig_L0 = sig_L0_func(h_nu[i])
		sig_L1 = sig_L1_func(h_nu[i])
		sig_W0 = sig_W0_func(h_nu[i])
		sig_W1 = sig_W1_func(h_nu[i])

		sig[i] = 1.0/(y + 1.0) * (sig_L0 + sig_W0) + (1.0 - 1.0 / (y + 1.0)) * (sig_L1 + sig_W1)
		
	return sig



#------- H2PhotoDissociationRate ----------------
def H2PhotoDissociationRate(h_nu, Jnu, sigH2, delta_nu):

	for i in range(len(h_nu)):

		fx = 4.0 * np.pi * Jnu[i] * sigH2[i] / h_nu_erg

	gH2 = 0.0

	for i in range(len(h_nu)-1):

		gH2 += delta_nu * (fx[i] + fx[i+1]) / 2.0
	
	return gH2



#------- A typical Eclipsing DLA Quasar -------
L_912 = 3.25e42 # erg/s/A
L_912 = flamb_to_fneu(L_912, 912.0) # L is now in erg/s/Hz
dist = 1e6 # 500.0 # pc
dist_cm = pc_to_cm(dist)
J912QSO = 1.0/(4. * np.pi) * L_912 / (4. * np.pi * dist_cm**2) # J at neu = 912 A. We assume a flat spectrum from 1 to 20 Ryd (Fathivavsari et al. 2015).
#----------------------------------------------


h_nu = np.arange(14.2, 17.5, 0.1)
h_nu_erg = [eV_to_erg(x) for x in h_nu]
hplanck = 6.626176e-27 # erg.s
nu = [x/hplanck for x in h_nu_erg]

delta_nu = nu[1] - nu[0]

Jnu = [J912QSO for x in h_nu_erg] # NOTE that we assume flat spectrum !!
sigH2 = sig_H2_photo_dissociation(h_nu)

P_LW = H2PhotoDissociationRate(h_nu, Jnu, sigH2, delta_nu)


print(f'P_LW = {P_LW}')

#*************************************************************************
#------------------ CALCULATION OF Q_HI, and Q_HeI -----------------------
#*************************************************************************

# E is the photon energy in eV.
E_H0 = 13.60 # eV
E_He0 = 24.59 # eV

neuH0 = np.arange(eV_to_neu(E_H0)/1e16, 6.0, 0.02) * 1e16
neuHe0 = np.arange(eV_to_neu(E_He0)/1e16, 6.0, 0.02) * 1e16

sigH0  = phCrossSectionX(neuH0, 4.298E-1, 5.475E4, 3.288E1, 2.963, 0.0, 0.0, 0.0) # H0
sigHe0 = phCrossSectionX(neuHe0, 1.361E+1, 9.492E2, 1.469E0, 3.188, 2.039, 4.434E-1, 2.136) # He0

JnuH0 = [J912QSO for x in neuH0]  # NOTE that we assume flat spectrum !!
JnuHe0= [J912QSO for x in neuHe0] # NOTE that we assume flat spectrum !!

fxH0 = np.zeros_like(neuH0)

for i in range(len(neuH0)):

	fxH0[i] = 4.0 * np.pi * JnuH0[i] * sigH0[i] / hplanck / neuH0[i]


fxHe0 = np.zeros_like(neuHe0)

for i in range(len(neuHe0)):

	fxHe0[i] = 4.0 * np.pi * JnuHe0[i] * sigHe0[i] / hplanck / neuHe0[i]


delta_nu_H0 = neuH0[1] - neuH0[0]
delta_nu_He0= neuHe0[1] - neuHe0[0]

gHI = 0.0

for i in range(len(neuH0) - 1):
	
	gHI += delta_nu_H0 * (fxH0[i] + fxH0[i+1]) / 2.0


gHeI = 0.0

for i in range(len(neuHe0) - 1):

	gHeI += delta_nu_He0 * (fxHe0[i] + fxHe0[i + 1]) / 2.0


P_HI = gHI
P_HeI= gHeI

print(f'P_HI = {P_HI}')
print(f'P_HeI = {P_HeI}')







