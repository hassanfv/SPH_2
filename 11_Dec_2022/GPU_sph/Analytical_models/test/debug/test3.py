
import numpy as np
import matplotlib.pyplot as plt
import readchar


f_c = 0.16
f_g = 0.016
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


R_kpc = 10

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
xi = xsi(L, n, R_kpc)
print(f'For L = {L:.3E}, n(r) = {n}, and r = {R_kpc} kpc, the ionization parameter is: {round(xi, 2)}')


def dEdt(n, T, xsi):

	a = -18./(np.exp(25.*(np.log10(T) - 4.35)**2)) - 80./(np.exp(5.5*(np.log10(T) - 5.2)**2)) - 17./(np.exp(3.6*(np.log10(T) - 6.5)**2))
	b = 1.7e4 / T**(0.7)
	c = 1.1 - 1.1/np.exp(T/1.8e5) + 4e15/T**4
	
	xsi_0 = 1./(1.5/np.sqrt(T) + 1.5e12/np.sqrt(T**5)) + 4e10/T**2 * (1. + 80./(np.exp(T - 1e4)/1.5e3))
	
	S1 = -3.8e-27 * np.sqrt(T) # the bremsstrahlung losses
	S2 = 4.1e-35 * (1.9e7 - T) * xsi # the Compton heating/cooling
	S3 = 1e-23 * (a + b * (xsi / xsi_0)**c) / ((1. + (xsi / xsi_0)**c)) # the sum of photoionization heating, line and recombination cooling.
	
	E_dot = n**2 * (S1 + S2 + S3)
	
	return E_dot

#oneMyr_in_sec = 1e6 * 365.25 * 24 * 3600
dt = 50.0 * 365.25 * 24 * 3600 # 50 years

T = T_shock_ISM(f, sig) # K

mu = 0.6 # fully ionized gas
rho_gas = mu * mH * n

E_0 = 3.*kB*rho_gas*T / 2/mu/mH
print(E_0)
E_previous = E_0

print(E_0)

t = 0

tEnd = 2e6 * 365.25 * 24 * 3600

res = []

R_kpc = 1.0 # starting radius

while R_kpc <= 10:


	R_kpc = 10

	t = 0
	
	T = T_shock_ISM(f, sig)
	
	t_dyn_sec = dynamical_time(R_kpc, sig)
	t_dyn_yr = t_dyn_sec / 3600. / 24. / 365.25

	while t < t_dyn_sec:

		n = n_density(f, sig, R_kpc)
		xi = xsi(L, n, R_kpc)
		rho_gas = mu * mH * n

		print(f'Input to E_dot = {n}, {T:.3E}, {xi}')
		print(f'T_shock = {T_shock_ISM(f, sig):.3E}, {R_kpc}')
		E_dot = dEdt(n, T, xi)
		E_new = E_previous + E_dot * dt
		
		E_previous = E_new
		
		T = 2.*mu*mH*E_new / 3./kB/rho_gas
		

		print(f'{t/365.25/24/3600:.3E}, {T:.3E}, {E_dot}, {E_previous}, {E_new}')
		print()
		
		if T < 1.1e4:
			T = 1.1e4
		
		t += dt
		
		#kb = readchar.readkey()
	
		#if kb == 'q':
		#	break
	
	res.append([R_kpc, T])
	#print(f'{R_kpc}, {T}, {t_dyn_yr:.3E}, {n}, {xi}')
	
	R_kpc += 1


res = np.array(res)

R = res[:, 0]
T = res[:, 1]

print(res)

plt.scatter(np.log10(R), T, s = 1, color = 'black')

#plt.xlim(0.1, 100)
plt.ylim(1e4, 1e9)

plt.yscale('log')

plt.show()









