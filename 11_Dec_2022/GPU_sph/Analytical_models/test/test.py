
import numpy as np


f_c = 0.16
f_g = 0.16
f = f_g / f_c
sig = 200 # km/s
sig = sig * 1000. * 100. # cm/s
c = 29979245800.0 # cm/s
G = 6.6738e-8
mH = 1.6726e-24
kB = 1.3807e-16

L = 4. * f_c * sig**4 * c / G
print(f'L = {L:.3E} erg/s')


def n_density(f, sig, R_kpc):

	sig_200 = sig / 200. / 1000. / 100.
	n = 60. * f * sig_200**2 / R_kpc**2
	
	return n


R_kpc = 1.0

print(f'rho at {R_kpc} kpc = {n_density(f, sig, R_kpc)} cm^-3')


def T_shock_ISM(f, sig):

	sig_200 = sig / 200. / 1000. / 100.
	T_sh = 2.2e7 * sig_200**(4./3.) * f**(-2./3.)
	
	return T_sh

print(f'T_sh_ISM = {T_shock_ISM(f, sig):.3E} K')


def dynamical_time(R_kpc, sig):

	R_cm = R_kpc * 1000. *  3.086e18
	
	return R_cm/sig


t_dyn_sec = dynamical_time(R_kpc, sig)

t_dyn_yr = t_dyn_sec / 3600. / 24. / 365.25
print(f'At R_pkc = {R_kpc}, t_dyn = {t_dyn_yr:.3E} years')


# xsi is the ionization_parameter
def xsi(L, n, R_kpc):
	
	R_cm = R_kpc * 1000. *  3.086e18
	return L / n / R_cm**2

n = n_density(f, sig, R_kpc)
xsiT = xsi(L, n, R_kpc)
print(f'For L = {L:.3E}, n(r) = {n}, and r = {R_kpc} kpc, the ionization parameter is: {round(xsiT, 2)}')


#------- Test
R_cm = R_kpc * 1000. *  3.086e18
mu = 0.6 # fully ionized gas
rho_gas = mu * mH * n
rho_e = rho_gas
xi = 3. * L / rho_e / R_cm**2
print(f'xi = {xi}')
#-----------------------------

def dEdt(n, T, xsi):

	a = -18./(np.exp(25.*(np.log10(T) - 4.35)**2)) - 80./(np.exp(5.5*(np.log10(T) - 5.2)**2)) - 17./(np.exp(3.6*(np.log10(T) - 6.5)**2))
	b = 1.7e4 / T**(0.7)
	c = 1.1 - 1.1/np.exp(T/1.8e5) + 4e15/T**4
	
	xsi_0 = 1./(1.5/np.sqrt(T) + 1.5e12/np.sqrt(T**5)) + 4e10/T**2 * (1. + 80./(np.exp(T - 1e4)/1.5e3))
	
	S1 = -3.8e-27 * np.sqrt(T) # the bremsstrahlung losses
	S2 = 4.1e-35 * (1.9e7 - T) * xsi # the Compton heating/cooling
	S3 = 1e-23 * (a + b * (xsi / xsi_0)**c) / ((1. + (xsi / xsi_0)**c)) # the sum of photoionization heating, line and recombination cooling.
	
	E_dot = n**2 * (S1 + S2 + S3)
	
	print('xsi_0 = ', xsi_0)
	
	return E_dot


T = 1e5

E_dot = dEdt(n, T, xsiT)

print(f'dEdt = {E_dot}')







