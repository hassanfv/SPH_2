
# DO NOT USE version before v3. In this version(i.e. v3) we fixed "Abundance_hX", "convert_u_to_temp_h" and "convert_Temp_to_u" functions!!
# The difference with photolibs.py is that here we include "convert_Temp_to_u" function.

import numpy as np

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



#===== phCrossSectionX (Verner et al - 1996) ===> Here we use a "for loop" instead of array operations!!
def phCrossSectionX(neu, E0, sig0, ya, P, yw, y0, y1):

	N = len(neu)
	sig = np.zeros(N)

	for i in range(N):
		
		E = neu_to_eV(neu[i])
		
		x = E/E0 - y0
		y = np.sqrt(x**2 + y1**2)
		Fy = (((x - 1)**2 + yw**2) * y**(0.5*P - 5.5)) * (1.0 + np.sqrt(y/ya))**(-P)
		
		sig[i] = sig0 * Fy * 1e-18
	
	return sig



#===== RandCIRates (Recombination and Collisional Ionization Rates)
def RandCIRates(T):

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
	
	dne = 1.0 # chosen just to be larger than 1e-4 so that we can go inside while loop !

	while((dne > 1e-4) and (niter < MAXITER)):

		ne_old = ne

		nH0 = aHp / (aHp + geH0 + gJH0 / (ne * nHcgs))

		nHp = 1.0 - nH0

		nHep = y / (1.0 + (aHep + ad) / (geHe0 + gJHe0/(ne * nHcgs)) + (geHep + gJHep/(ne * nHcgs)) / aHepp)

		nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0/(ne * nHcgs))

		nHepp = nHep * (geHep + gJHep/(ne * nHcgs)) / aHepp

		ne = nHp + nHep + 2.0 * nHepp
		
		ne_new = 0.5 * (ne + ne_old)
		ne = ne_new
		dne = np.abs(ne - ne_old)
		
		niter += 1
		
	return nH0, nHe0, nHp, ne, nHep, nHepp




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
	Jcoeff = 0.00001

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

	sigH0  = phCrossSectionX(neuH0, 4.298E-1, 5.475E4, 3.288E1, 2.963, 0.0, 0.0, 0.0) # H0
	sigHe0 = phCrossSectionX(neuHe0, 1.361E+1, 9.492E2, 1.469E0, 3.188, 2.039, 4.434E-1, 2.136) # He0
	sigHep = phCrossSectionX(neuHep, 1.720E+0, 1.369E4, 3.288E1, 2.963, 0.0, 0.0, 0.0) # Hep

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



#===== coolinHeatingRates (For your desired Radiation Field, please modify the 'RadiationField' function.)
def coolingHeatingRates(T, nHcgs): 

	gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField() # Note: we must multiply these values by the nH0, nHe0, and nHep
									       # to make them heating Rate by photoionization.
	aHp, aHep, aHepp, ad, geH0, geHe0, geHep = RandCIRates(T)
	nH0, nHe0, nHp, ne, nHep, nHepp = Abundance_hX(T, nHcgs, gJH0, gJHe0, gJHep)

	HeatingRate_H0 = nH0 * HRate_H0 / nHcgs # This is H/nH^2
	HeatingRate_He0 = nHe0 * HRate_He0 / nHcgs
	HeatingRate_Hep = nHep * HRate_Hep / nHcgs

	Total_HeatingRate = HeatingRate_H0 + HeatingRate_He0 + HeatingRate_Hep

	#************* COOLING SECTION **********************
	Tfact = 1.0 / (1.0 + np.sqrt(T/1e5))
	# collisional excitation (Cen 1992):
	BetaH0 = 7.5e-19 * np.exp(-118348/T) * Tfact
	BetaHep = 5.54e-17 * T**(-0.397) * np.exp(-473638/T) * Tfact
	# free-free:
	Betaff = 1.43e-27 * np.sqrt(T) * ( 1. + 0.34 * np.exp(-(5.5 - np.log10(T))**2 / 3) )

	LambdaExcH0   = BetaH0 * ne * nH0
	LambdaExcHep  = BetaHep * ne * nHep
	LambdaExc     = LambdaExcH0 + LambdaExcHep # /* excitation */

	LambdaIonH0   = 2.18e-11 * geH0 * ne * nH0
	LambdaIonHe0  = 3.94e-11 * geHe0 * ne * nHe0
	LambdaIonHep  = 8.72e-11 * geHep * ne * nHep
	LambdaIon     = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep # /* ionization */

	LambdaRecHp   = 1.036e-16 * T * ne * (aHp * nHp)
	LambdaRecHep  = 1.036e-16 * T * ne * (aHep * nHep)
	LambdaRecHepp = 1.036e-16 * T * ne * (aHepp * nHepp)
	LambdaRecHepd = 6.526e-11 * ad * ne * nHep
	LambdaRec     = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd

	LambdaFF      = Betaff * (nHp + nHep + 4 * nHepp) * ne

	Lambda        = LambdaExc + LambdaIon + LambdaRec + LambdaFF
	#----------------------------------------------------

	return Total_HeatingRate, Lambda # note that Total_HeatingRate is HeatingRate/nH^2 and Lambda is Lambda/nH^2.
	#return 0.0, Lambda # note that Total_HeatingRate is HeatingRate/nH^2 and Lambda is Lambda/nH^2. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#===== convert_u_to_temp_h
def convert_u_to_temp_h(u, nHcgs, XH):
	
	# u MUST be in physical units !!!!!!
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH = 1.6726e-24 # gram
	gamma = 5.0/3.0
	
	yHelium = (1.0 - XH) / (4. * XH)

	ne_guess = 1.0 # our initial guess is that elec_density = hydrogen density.
	mu = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess) # yHelium = nHe/nH and ne = ne/nH
	temp = (gamma - 1.0) * mH / kB * mu * u
	
	MAXITER = 100
	temp_old = temp/2.
	niter = 1
	
	gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField()
	
	dTemp = temp
	
	while ((dTemp/temp > 1e-4) and (niter < MAXITER)):

		temp_old = temp

		# updating ne
		nH0, nHe0, nHp, ne_guess, nHep, nHepp = Abundance_hX(temp_old, nHcgs, gJH0, gJHe0, gJHep)

		mu = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess)
		temp = (gamma - 1.0) * mH / kB * mu * u

		new_temp = 0.5 * (temp + temp_old)
		temp = new_temp
		dTemp = np.abs(temp - temp_old)

		niter += 1

	if niter >= MAXITER:
		print('convert_u_to_temp_h Failed to converge !')
	
	return temp




#===== convert_u_to_temp_h
def convert_Temp_to_u(temp, nHcgs, XH):
	
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH = 1.6726e-24 # gram
	gamma = 5.0/3.0
	
	yHelium = (1.0 - XH) / (4. * XH)

	ne_guess = 1.0 # our initial guess is that elec_density = hydrogen density.
	mu = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess) # yHelium = nHe/nH and ne = ne/nH

	u = kB/mH/(gamma - 1.0)/mu * temp # Navarro & White 1993.
	
	MAXITER = 100
	u_old = u/2.
	niter = 1
	
	du = u
	
	while ((du/u > 1e-4) and (niter < MAXITER)):

		u_old = u

		# updating ne
		gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField()
		temp_old = convert_u_to_temp_h(u_old, nHcgs, XH)
		nH0, nHe0, nHp, ne_guess, nHep, nHepp = Abundance_hX(temp_old, nHcgs, gJH0, gJHe0, gJHep)

		mu = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess)
		u = kB/mH/(gamma - 1.0)/mu * temp
		
		new_u = 0.5 * (u + u_old)
		u = new_u
		du = np.abs(u - u_old)

		niter += 1
	
	if niter >= MAXITER:
		print('convert_Temp_to_u Failed to converge !')
	
	return u



#===== coolingRateFromU_h
def coolingRateFromU_h(u, nHcgs, XH):

	# Note that u must be in physical units (i.e. cgs);
	
	Tmin = np.log10(1e4)
	Tmax = np.log10(1e8)
	
	Temp = convert_u_to_temp_h(u, nHcgs, XH)
	
	if Temp < 10**Tmin: Temp = 10**Tmin
	if Temp > 10**Tmax: Temp = 10**Tmax
	
	Heat, Lambda = coolingHeatingRates(Temp, nHcgs)
	
	return Heat - Lambda



#===== DoCooling_h
def DoCooling_h(rho, u_old, dt, XH):
	
	MAXITER = 100
	mH = 1.6726e-24 # gram
	nHcgs    = XH * rho / mH # hydrogen number dens in cgs units 
	ratefact = nHcgs**2 / rho # The effect of the density on the cooling is here.

	u = u_old
	u_upper = u
	
	GammaLambdaNet = coolingRateFromU_h(u, nHcgs, XH)	
	
	if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0):
	
	
		while (u_upper - u_old - ratefact * coolingRateFromU_h(u_upper, nHcgs, XH) * dt > 0.0):
			u_upper /= 1.1
			
		u_lower = u_upper
		u_upper = u_lower * 1.1
	
	if (u - u_old - ratefact * GammaLambdaNet * dt < 0.0):
	
		while (u_upper - u_old - ratefact * coolingRateFromU_h(u_upper, nHcgs, XH) * dt < 0.0):
			u_upper *= 1.1
		
		u_lower = u_upper / 1.1
		u_upper = u_upper
	
	MAXITER = 100
	niter = 1
	du = u
	
	while ((np.abs(du/u) > 1e-4) & (niter < MAXITER)):
	
		u = 0.5 * (u_lower + u_upper)
		
		GammaLambdaNet = coolingRateFromU_h(u, nHcgs, XH)
		
		if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0):
			u_upper = u
		else:
			u_lower = u
		
		du = np.abs(u_upper - u_lower)
	
		niter += 1

	print(GammaLambdaNet, dt, ratefact)

	if niter >= MAXITER:
		print('Failed to converge !')
		
	return u # Note that this u is in physical units. We must convert it to code unit before using it in our main program. We can convert it 
		 # to the code unit by by "multiplying" it by UnitDensity_in_cgs and then "dividing" by UnitPressure_in_cgs






