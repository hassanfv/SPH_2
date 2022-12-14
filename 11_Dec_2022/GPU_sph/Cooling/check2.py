
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#===== lamb_to_neu
def lamb_to_neu(lamb):

	clight = 29979245800.0 # cm/s
	lamb_in_cm = lamb * 1e-8
	neu = clight / lamb_in_cm
	
	return neu


#===== neu_to_lamb
def neu_to_lamb(neu):

	clight = 29979245800.0 # cm/s
	lamb = clight / neu * 1e8
	
	return lamb


#===== eV_to_neu
def eV_to_neu(eV):

	h_in_eV = 4.135667696e-15 # h in eV.Hz^-1
	neu = eV / h_in_eV

	return neu


#===== neu_to_eV
def neu_to_eV(neu):

	h_in_eV = 4.135667696e-15 # h in eV.Hz^-1
	eV = h_in_eV * neu
	
	return eV


#===== eV_to_lamb
def eV_to_lamb(eV):

	neu = eV_to_neu(eV)
	lamb = neu_to_lamb(neu)
	
	return lamb


#===== lamb_to_eV
def lamb_to_eV(lamb):

	neu = lamb_to_neu(lamb)
	eV = neu_to_eV(neu)
	
	return(eV)


#===== ryd_to_eV
def ryd_to_eV(ryd):

	return 13.60 * ryd


#===== eV_to_ryd
def eV_to_ryd(eV):

	return eV / 13.60


#===== fneu_to_flamb
def fneu_to_flamb(fneu, neu):

	lamb = neu_to_lamb(neu)
	flamb = 2.99792458e18 / lamb**2 * fneu

	return flamb


#===== flamb_to_fneu
def flamb_to_fneu(flamb, lamb):

	neu = lamb_to_neu(lamb)
	fneu = 2.99792458e18 / neu**2 * flamb
	
	return fneu


#===== pc_to_cm
def pc_to_cm(pc):

	return pc * 3.086e+18


#===== cm_to_pc
def cm_to_pc(cm):

	return cm / 3.086e+18


#===== phCrossSection (Verner et al - 1996)
def phCrossSection(neu, E0, sig0, ya, P, yw, y0, y1):

	E = np.array([neu_to_eV(neut) for neut in neu]) # photon energy in eV.

	x = E/E0 - y0
	y = np.sqrt(x**2 + y1**2)
	Fy = (((x - 1)**2 + yw**2) * y**(0.5*P - 5.5)) * (1.0 + np.sqrt(y/ya))**(-P)

	sig = sig0 * Fy * 1e-18

	return sig


def J_neu_func(neu, eV, Jcoeff, J912QSO, RFiD):
	
	#------- J_neu from Vedel et al. 1994 ---------
	#z_redshift = 4.0
	J_minus_21 = Jcoeff  # 0.1
	
	Jneu = J_minus_21 * 1e-21 / (neu/lamb_to_neu(eV_to_lamb(eV)))
	
	if RFiD == 'QSO':
		return J912QSO + 0.0 * neu
	else:
		return Jneu




#===== RadiationField
def RadiationField():

	RFiD = 'NotQSO' # if you want J912 of a quasar set this to 'QSO'.
	Jcoeff = 0.1

	#------- A typical Eclipsing DLA Quasar -------
	L_912 = 3.25e42 # erg/s/A
	L_912 = flamb_to_fneu(L_912, 912.0) # L is now in erg/s/Hz
	dist = 500.0 # pc
	dist_cm = pc_to_cm(dist)
	J912QSO = 1.0/(4. * np.pi) * L_912 / (4. * np.pi * dist_cm**2) # J at neu = 912 A. We assume a flat spectrum from 1 to 20 Ryd (Fathivavsari et al. 2015).
	#----------------------------------------------

	# E is the photon energy in eV.
	E_H0 = 13.60 # eV
	E_He0 = 24.59 # eV
	E_Hep = 54.42 # eV

	neuH0 = np.arange(eV_to_neu(E_H0)/1e16, 6.0, 0.02) * 1e16
	neuHep = np.arange(eV_to_neu(E_Hep)/1e16, 6.0, 0.02) * 1e16
	neuHe0 = np.arange(eV_to_neu(E_He0)/1e16, 6.0, 0.02) * 1e16

	sigH0  = phCrossSection(neuH0, 4.298E-1, 5.475E4, 3.288E1, 2.963, 0.0, 0.0, 0.0) # H0
	sigHe0 = phCrossSection(neuHe0, 1.361E+1, 9.492E2, 1.469E0, 3.188, 2.039, 4.434E-1, 2.136) # He0
	sigHep = phCrossSection(neuHep, 1.720E+0, 1.369E4, 3.288E1, 2.963, 0.0, 0.0, 0.0) # Hep

	#***********************************************************************************************************
	#***************************************** Photoionization rate ********************************************
	#***********************************************************************************************************

	hplanck = 6.626196E-27 # h in erg.Hz^-1 or erg.s

	J_neu = J_neu_func(neuH0, 13.60, Jcoeff, J912QSO, RFiD)
	fxH0 = 4. * np.pi * J_neu * sigH0 / hplanck / neuH0

	J_neu = J_neu_func(neuHe0, 24.59, Jcoeff, J912QSO, RFiD)
	fxHe0 = 4. * np.pi * J_neu * sigHe0 / hplanck / neuHe0

	J_neu = J_neu_func(neuHep, 54.42, Jcoeff, J912QSO, RFiD)
	fxHep = 4. * np.pi * J_neu * sigHep / hplanck / neuHep

	#----- H0 Photoionization Rate ------------
	delta_neu_H0 = neuH0[1] - neuH0[0]

	gJH0 = 0.0
	for i in range(len(sigH0)-1):

		gJH0 += delta_neu_H0 * (fxH0[i] + fxH0[i+1]) / 2.

	#----- He0 Photoionization Rate ------------
	delta_neu_He0 = neuHe0[1] - neuHe0[0]
	gJHe0 = 0.0
	for i in range(len(sigHe0)-1):

		gJHe0 += delta_neu_He0 * (fxHe0[i] + fxHe0[i+1]) / 2.

	#----- Hep Photoionization Rate ------------
	delta_neu_Hep = neuHep[1] - neuHep[0]
	gJHep = 0.0
	for i in range(len(sigHep)-1):

		gJHep += delta_neu_Hep * (fxHep[i] + fxHep[i+1]) / 2.

	#***********************************************************************************************************
	#***************************************** Heating rate ****************************************************
	#***********************************************************************************************************

	J_neu = J_neu_func(neuH0, 13.60, Jcoeff, J912QSO, RFiD)
	fxH0 = 4. * np.pi * J_neu * sigH0 * (neuH0 - eV_to_neu(E_H0)) / neuH0

	J_neu = J_neu_func(neuHe0, 24.59, Jcoeff, J912QSO, RFiD)
	fxHe0 = 4. * np.pi * J_neu * sigHe0 * (neuHe0 - eV_to_neu(E_He0)) / neuHe0

	J_neu = J_neu_func(neuHep, 54.42, Jcoeff, J912QSO, RFiD)
	fxHep = 4. * np.pi * J_neu * sigHep * (neuHep - eV_to_neu(E_Hep)) / neuHep

	#----- H0 Heating Rate ------------
	delta_neu_H0 = neuH0[1] - neuH0[0]
	HRate_H0 = 0.0
	for i in range(len(sigH0)-1):

		HRate_H0 += delta_neu_H0 * (fxH0[i] + fxH0[i+1]) / 2.

	#----- He0 Heating Rate ------------
	delta_neu_He0 = neuHe0[1] - neuHe0[0]
	HRate_He0 = 0.0
	for i in range(len(sigHe0)-1):

		HRate_He0 += delta_neu_He0 * (fxHe0[i] + fxHe0[i+1]) / 2.

	#----- Hep Heating Rate ------------
	delta_neu_Hep = neuHep[1] - neuHep[0]
	HRate_Hep = 0.0
	for i in range(len(sigHep)-1):

		HRate_Hep += delta_neu_Hep * (fxHep[i] + fxHep[i+1]) / 2.

	return gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep




#===== Abundance_hX
def Abundance_hX(T, nHcgs, gJH0, gJHe0, gJHep):

	Tfact = 1.0 / (1.0 + np.sqrt(T/1e5))

	aHp, aHep, aHepp, ad, geH0, geHe0, geHep = RandCIRates(T)

	Y = 0.24 # Helium abundance by mass.
	y = Y / (4.0 - 4.0 * Y)

	# NOTE: all number densities are relative to nH.

	ne = 1.0 # initial guess
	ne_old = ne / 2.
	
	MAXITER = 100
	niter = 1

	while((np.abs(ne - ne_old) > 1e-2 * ne) and (niter < MAXITER)):

		ne_old = ne

		nH0 = aHp / (aHp + geH0 + gJH0 / (ne * nHcgs))

		nHp = 1.0 - nH0

		nHep = y / (1.0 + (aHep + ad) / (geHe0 + gJHe0/(ne * nHcgs)) + (geHep + gJHep/(ne * nHcgs)) / aHepp)

		nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0/(ne * nHcgs))

		nHepp = nHep * (geHep + gJHep/(ne * nHcgs)) / aHepp

		ne = nHp + nHep + 2.0 * nHepp
		
		niter += 1
		
	return nH0, nHe0, nHp, ne, nHep, nHepp



#===== RandCIRates (Recombination and Collisional Ionization Rates)
def RandCIRates(T): # T in Kelvin.

	# ****** Recombination and Collisional Ionization Rates ************
	Tfact = 1.0 / (1.0 + np.sqrt(T/1e5))

	# recombination (Cen 1992):
	# Hydrogen II:
	AlphaHp = 8.41e-11 * (T/1000.0)**(-0.2) / (1. + (T/1e6)**(0.7)) / np.sqrt(T)
	# Helium II:
	AlphaHep = 1.5e-10 * T**(-0.6353)
	# Helium III:
	AlphaHepp = 4. * AlphaHp

	# dielectric recombination
	Alphad = 1.9e-3 * T**(-1.5) * np.exp(-470000.0/T) * (1. + 0.3 * np.exp(-94000.0/T))

	# collisional ionization (Cen 1992):
	# Hydrogen:
	GammaeH0   = 5.85e-11 * np.sqrt(T) * np.exp(-157809.1/T) * Tfact
	# Helium:
	GammaeHe0  = 2.38e-11 * np.sqrt(T) * np.exp(-285335.4/T) * Tfact
	# Helium II:
	GammaeHep  = 5.68e-12 * np.sqrt(T) * np.exp(-631515.0/T) * Tfact
	#*******************************************************************

	return AlphaHp, AlphaHep, AlphaHepp, Alphad, GammaeH0, GammaeHe0, GammaeHep


Tgrid = np.logspace(4, 8, 100)

#gJH0 = gJHe0 = gJHep = 0.0

gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField()

print(gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep)

nHcgs = 1e-2 # cm^-3

res = []

for T in Tgrid:
	nH0, nHe0, nHp, ne, nHep, nHepp = Abundance_hX(T, nHcgs, gJH0, gJHe0, gJHep)
	res.append([nH0, nHe0, nHp, ne, nHep, nHepp])

res = np.array(res)

nH0   = res[:, 0]
nHe0  = res[:, 1]
nHp   = res[:, 2]
ne    = res[:, 3]
nHep  = res[:, 4]
nHepp = res[:, 5] 


plt.plot(Tgrid, nH0, color = 'black', label = 'nH0')
plt.plot(Tgrid, nHp, color = 'grey', label = 'nHp')
plt.plot(Tgrid, nHe0, color = 'blue', label = 'nHe0')
plt.plot(Tgrid, nHep, color = 'purple', label = 'nHep')
plt.plot(Tgrid, nHepp, color = 'lime', label = 'nHepp')
plt.plot(Tgrid, ne, color = 'red', label = 'ne')

plt.xscale('log')
plt.yscale('log')

plt.ylim(1e-2, 2)

plt.legend()
plt.show()







