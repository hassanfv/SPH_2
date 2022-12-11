
import numpy as np
import matplotlib.pyplot as plt
import time

h = 0.1 # used for numerical differentiation

X = 0.75
Y = 0.25
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
theta_rot = 85.4 # K
theta_vib = 6100. # K
R_0 = kB / mH
gamma = 5.0/3.0

D_0 = 4.477 # eV  Note that when we use it below, we convert it to erg by multiplying it to 1.602e-12

E0 = 0.0
E1 = 1.185e-14
E2 = 6.0 * E1
E3 = 12.0 * E1

E10 = 5860. * kB
E20 = 2. * E10

#***** z_para
def z_para(T):

	NoP = 10
	s = 0.
	for j in range(0, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return s


#***** ln_z_p
def ln_z_p(T):

	NoP = 10
	s = 0.
	for j in range(0, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return np.log(s)


#***** z_ortho
def z_ortho(T):

	NoP = 10
	s = 0.
	for j in range(1, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return s


#***** ln_z_o
def ln_z_o(T):

	NoP = 10
	s = 0.
	for j in range(1, NoP, 2):

		s += (2.*j+1.) * np.exp(-j*(j+1.)*theta_rot/T)

	return np.log(s)


#***** deriv_h2
def deriv_h2(func, x, h):

	df = (func(x+h) - func(x-h)) / 2. / h

	return df


#***** E_H2_func
def E_H2_func(ln_z_p, ln_z_o, T, h):

	f_p = T*T * deriv_h2(ln_z_p, T, h)
	f_o = T*T * deriv_h2(ln_z_o, T, h)
	f_tot = (z_para(T) * f_p + 3. * z_ortho(T) * f_o) / (z_para(T) + 3. * z_ortho(T))

	E_H2 = kB * (3./2. * T + f_tot + theta_vib / (np.exp(theta_vib/T) - 1.))

	return E_H2



#***** gammaH
def gammaH(J, T3): # It seems it should be T instead of T3. May be the original paper made a typo mistake !!!! Investigate later. 
	
	tmp1 = 1e-11 * T3**0.5 / (1. + 60./T3**4) + 1e-12 * T3
	tmp2 = 0.33 + 0.9 * np.exp(-1.*((J - 3.5)/0.9)**2)
	
	return tmp1 * tmp2


#***** gammaH2
def gammaH2(J, T3): # It seems it should be T instead of T3. May be the original paper made a typo mistake !!!! Investigate later. 
	
	return (3.3e-12 + 6.6e-12 * T3) * (0.276 * J*J * np.exp(-(J/3.18)**1.7))


#***** cooling_rate
def cooling_rate(T, nHI, nH2, n_tot):

	nHI = nHI + 0.0001 # to avoid division by zero
	nH2 = nH2 + 0.0001

	T3 = T / 1000.0

	#********* Beginning of H section ********
	#-------------- L_r_H -----------------
	L_r_H_1 = (9.5e-22 *T3**3.76/(1.+0.12*T3**2.1)) * np.exp(-1.*(0.13/T3)**3)
	L_r_H_2 = 3e-24 * np.exp(-0.51/T3)
	L_r_H = 1./nHI * (L_r_H_1 + L_r_H_2)

	L_r_H_n_0_1 = 5. * gammaH(2, T) * np.exp(-1.*(E2 - E0)/kB/T) * (E2 - E0)
	L_r_H_n_0_2 = 7./3. * gammaH(3, T) * np.exp(-1.*(E3 - E1)/kB/T) * (E3 - E1)
	L_r_H_n_0 = 0.25*L_r_H_n_0_1 + 0.75*L_r_H_n_0_2

	nHrot_nHI = L_r_H / L_r_H_n_0

	H_rot = L_r_H / (1. + nHrot_nHI)
	#----------------


	L_v_H = 1./nHI * (6.7e-19 * np.exp(-5.86/T3) + 1.6e-18 * np.exp(-11.7/T3))
	L_v_H_n_0 = gammaH(10, T) * np.exp(-E10/kB/T) * E10 + gammaH(20, T) * np.exp(-E20/kB/T) * E20

	nHvib_nHI = L_v_H / L_v_H_n_0

	H_vib = L_v_H / (1. + nHvib_nHI)

	LH = H_rot + H_vib
	#******* End of H section ********
	
	
	#******** Begining of H2 section **********
	L_r_H2_1 = (9.5e-22 *T3**3.76/(1.+0.12*T3**2.1)) * np.exp(-1.*(0.13/T3)**3)
	L_r_H2_2 = 3e-24 * np.exp(-0.51/T3)
	L_r_H2 = 1./nH2 * (L_r_H2_1 + L_r_H2_2)
	
	L_r_H2_n_0_1 = 5. * gammaH2(2, T) * np.exp(-1.*(E2 - E0)/kB/T) * (E2 - E0)
	L_r_H2_n_0_2 = 7./3. * gammaH2(3, T) * np.exp(-1.*(E3 - E1)/kB/T) * (E3 - E1)
	L_r_H2_n_0 = 0.25*L_r_H2_n_0_1 + 0.75*L_r_H2_n_0_2
	
	nH2rot_nH2 = L_r_H2 / L_r_H2_n_0
	
	H2_rot = L_r_H2 / (1. + nH2rot_nH2)
	#----------------
	
	L_v_H2 = 1./nH2 * (6.7e-19 * np.exp(-5.86/T3) + 1.6e-18 * np.exp(-11.7/T3))
	L_v_H2_n_0 = gammaH2(10, T) * np.exp(-E10/kB/T) * E10 + gammaH2(20, T) * np.exp(-E20/kB/T) * E20
	
	nH2vib_nH2 = L_v_H2 / L_v_H2_n_0
	
	H2_vib = L_v_H2 / (1. + nH2vib_nH2)
	
	LH2 = H2_rot + H2_vib
	#******* End of H2 section ********
	

	#******** Beginning of CO section **********
	
	N_rat = (3.3e6 * T3**0.5)/n_tot
	L_CO = 4.*(kB*T)**2 * 9.7e-8 / (2.76*n_tot*kB * (1. + N_rat + 1.5 * N_rat**0.5))
	
	if T < 5.:
		L_CO = 0.0

	#******* End of CO section *****************
	
	
	#******** Beginning of HII section **********
	
	if np.log10(T) <= 6.2:
		thetaT = 0.2 * (6.2 - np.log10(T))**4
	else:
		thetaT = 0.0
	
	L_HII = 1e-21 * (10**(-0.1-1.88*(5.23-np.log10(T))**4 + 10**(-1.7-thetaT)))
	
	#******* End of HII section ******************
	
	Lambda_CO = 8.5e-5 * n_tot*n_tot * L_CO
	Lambda_H2 = nH2 * nH2 * LH2
	Lambda_HI = nHI * nH2 * LH
	Lambda_HII = nHI * nHI * L_HII
	
	Lambda = Lambda_CO + Lambda_H2 + Lambda_HI + Lambda_HII

	return Lambda # it will be multiplied by -1 in the coolingRateFromU_M function.



#***** h1_h2_abundance
def h1_h2_abundance(T, rho):  # !!!!!!!!!!!!!!!!!!!!!!!!1 THERE IS A CONCERN HERE !!!!!! Read the comment on the next line !!!!!!!

	max_nH = rho / mH * (X + Y/4.) # We need this because nHI and nH2 behave very strangely at T ~ 4000 K. Please investigate it so that
				       # we do not need use max_nH to circumvent the problem...

	K = 2.11 / rho / X * np.exp(-52490./T)

	delta = K*K + 4. * K

	y = (-K + delta**0.5) / 2.

	nHI = rho * X / mH * y
	
	if nHI > max_nH:
		nHI = max_nH
	
	nH2 = rho * X / (2.*mH) * (1. - y)
	
	if nH2 < 0.0:
		nH2 = 0.0

	return nHI, nH2, nHI + 2. * nH2  # n_tot = nHI + 2. * nH2 # ML (1991)



#***** u_as_a_func_of_T
def u_as_a_func_of_T(T, rho):

	y = y_func(T, rho)

	mu_inverse = X/2. + X*y/2. + Y/4.

	f_p = T*T * deriv_h2(ln_z_p, T, h)
	f_o = T*T * deriv_h2(ln_z_o, T, h)
	f_tot = (z_para(T) * f_p + 3. * z_ortho(T) * f_o) / (z_para(T) + 3. * z_ortho(T))

	E_H2 = kB * (3./2. * T + f_tot + theta_vib / (np.exp(theta_vib/T) - 1.))

	uT = 3./2. * R_0 * mu_inverse * T + 1./2. * X * y * D_0 * 1.602e-12 / mH + X * (1. - y) * E_H2 / 2. / mH # 1.6e-12 converts D_0 from eV to erg.
	
	return uT


#***** y_func
def y_func(T, rho):

	K = 2.11 / rho / X * np.exp(-52490./T)

	delta = K*K + 4. * K

	y = (-K + delta**0.5) / 2.
	
	return y


#===== coolingRateFromU_M
def coolingRateFromU_M(u, T_init, rho):

	# Note that u must be in physical units (i.e. cgs);
	
	Tmin = np.log10(5)
	Tmax = np.log10(1e6)
	
	Temp = u_to_temp(u, T_init, rho)
	
	if Temp < 10**Tmin: Temp = 10**Tmin
	if Temp > 10**Tmax: Temp = 10**Tmax
	
	nHI, nH2, n_tot = h1_h2_abundance(Temp, rho)
	
	Lambda = cooling_rate(Temp, nHI, nH2, n_tot)
	
	return -Lambda



#***** u_to_temp
def u_to_temp(u, T_init, rho):

	T = T_init
	T_upper = T

	if u_as_a_func_of_T(T, rho) - u < 0.:

		while u_as_a_func_of_T(T_upper, rho) - u < 0.0:
			T_upper *= 1.1
		
		T_lower = T_upper / 1.1
		T_upper = T_upper


	if u_as_a_func_of_T(T, rho) - u > 0.:

		while u_as_a_func_of_T(T_upper, rho) - u > 0.:
			T_upper /= 1.1
		
		T_lower = T_upper
		T_upper = T_lower * 1.1

	MAXITER = 100
	dT = T
	niter = 0

	while (dT/T > 1e-3) and (niter < MAXITER):

		T = 0.5 * (T_lower + T_upper)
		
		if u_as_a_func_of_T(T, rho) - u > 0.:
		
			T_upper = T
		else:
			T_lower = T
		
		dT = np.abs(T_upper - T_lower)
		niter += 1
		
		if niter >= MAXITER:
			print('Failed to converge the Temperature !!')

	return T



#===== DoCooling_M
def DoCooling_M(rho, u_old, dt):

	# NOte that rho, u_old, T_init and dt all should be in physical units.
	
	MAXITER = 100
	ratefact = 1./rho  # The effect of the density on the cooling is here.
	
	#---- calculating some rough T_init ----
	y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
	mu_approx = X/2. + X*y/2. + Y/4.
	T_init = (gamma - 1.0) * mH / kB * mu_approx * u_old
	#---------------------------------------

	u = u_old
	u_upper = u
	
	TG = time.time()
	GammaLambdaNet = coolingRateFromU_M(u, T_init, rho) # coolingRateFromU_M(u, nHcgs, XH) # !!!!! orrect ARGs
	print('TG = ', time.time() - TG)
	
	
	TB = time.time()
	if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0):
	
	
		while (u_upper - u_old - ratefact * coolingRateFromU_M(u_upper, T_init, rho) * dt > 0.0):
			u_upper /= 1.1
			
		u_lower = u_upper
		u_upper = u_lower * 1.1
	
	if (u - u_old - ratefact * GammaLambdaNet * dt < 0.0):
	
		while (u_upper - u_old - ratefact * coolingRateFromU_M(u_upper, T_init, rho) * dt < 0.0):
			u_upper *= 1.1
		
		u_lower = u_upper / 1.1
		u_upper = u_upper
	
	print('TB = ', time.time() - TB)
	
	MAXITER = 100
	niter = 1
	du = u
	
	while ((np.abs(du/u) > 1e-6) & (niter < MAXITER)):
	
		u = 0.5 * (u_lower + u_upper)
		
		T_init = (gamma - 1.0) * mH / kB * mu_approx * u # updating T_init with the last estimated u. Should make it faster.
		GammaLambdaNet = coolingRateFromU_M(u, T_init, rho)
		
		if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0):
			u_upper = u
		else:
			u_lower = u
		
		du = np.abs(u_upper - u_lower)
	
		niter += 1

	if niter >= MAXITER:
		print('Failed to converge !')
		
	return u # Note that this u is in physical units. We must convert it to code unit before using it in our main program. We can convert it 
		 # to the code unit by "multiplying" it by UnitDensity_in_cgs and then "dividing" by UnitPressure_in_cgs.




#unit_U = 9537129466.43166
#UnitTime_in_s = 3.16e13 # == 1Myr
#unit_rho = 1.5e-20 # g/cm3

#T = 10. # K

#rho = 1.0 * unit_rho
#u = 0.9 * unit_U

#u = u_as_a_func_of_T(T, rho)

#print(u)
#print(u/unit_U)

#T_exact = u_to_temp(u, T_init, rho)

#print(T_init)
#print(T_exact)

#s()

#--- Testing a single cooling -----

#rho = 1.0
#u = 2.
#dt = 0.01

#rho_cgs = rho * unit_rho
#u_cgs = u * unit_U
#dt_cgs = dt * UnitTime_in_s

#---- calculating some rough T_init ----
#y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
#mu_approx = X/2. + X*y/2. + Y/4.
#T_init = (gamma - 1.0) * mH / kB * mu_approx * u_cgs
#---------------------------------------


#u_after_cooling = DoCooling_M(rho_cgs, u_cgs, dt_cgs)

#print()
#print('u before cooling = ', u)
#print('u after cooling = ', u_after_cooling/unit_U)


#Lambdax = coolingRateFromU_M(u_cgs, T_init, rho_cgs)

#print(Lambdax)







