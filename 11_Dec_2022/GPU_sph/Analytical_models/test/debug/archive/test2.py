
import numpy as np
import matplotlib.pyplot as plt

def n_density(f, sig, R_kpc):

	sig_200 = sig / 200. / 1000. / 100.
	n = 60. * f * sig_200**2 / R_kpc**2
	
	return n


def T_shock_ISM(f, sig):

	sig_200 = sig / 200. / 1000. / 100.
	T_sh = 2.2e7 * sig_200**(4./3.) * f**(-2./3.)
	
	return T_sh


def dynamical_time(R_kpc, sig):

	R_cm = R_kpc * 1000. *  3.086e18
	
	return R_cm/sig


# xsi is the ionization_parameter
def xsi(L, n, R_kpc):
	
	R_cm = R_kpc * 1000. *  3.086e18
	return L / n / R_cm**2


def dEdt(n, T, xsi):

	a = -18./(np.exp(25.*(np.log10(T) - 4.35)**2)) - 80./(np.exp(5.5*(np.log10(T) - 5.2)**2)) - 17./(np.exp(3.6*(np.log10(T) - 6.5)**2))
	b = 1.7e4 / T**(0.7)
	c = 1.1 - 1.1/np.exp(T/1.8e5) + 4e15/T**4
	
	xsi_0 = 1./(1.5/np.sqrt(T) + 1.5e12/np.sqrt(T**5)) + 4e10/T**2 * (1. + 80./(np.exp(T - 1e4)/1.5e3))
	
	S1 = -3.8e-27 * np.sqrt(T) # the bremsstrahlung losses
	S2 = 4.1e-35 * (1.9e7 - T) * xsi # the Compton heating/cooling
	S3 = 1e-23 * (a + b * (xsi / xsi_0)**c) / ((1. + (xsi / xsi_0)**c)) # the sum of photoionization heating, line and recombination cooling.
	
	E_dotx = n**2 * (S1 + S2) # if T > 3e7 K
	
	if T > 3e7:
		return E_dotx
	else:
		return E_dot



f_c = 0.16
f_g = 0.016

R_kpc = 10

f = f_g / f_c
sig = 200 # km/s
sig = sig * 1000. * 100. # cm/s
c = 29979245800.0 # cm/s
G = 6.6738e-8
mH = 1.6726e-24
kB = 1.3807e-16

mu = 0.6

L = 4. * f_c * sig**4 * c / G


t_dyn_sec = dynamical_time(R_kpc, sig)


n = n_density(f, sig, R_kpc)
xi = xsi(L, n, R_kpc)

dt = 50.0 * 365.25 * 24 * 3600 # 50 years

T = T_shock_ISM(f, sig) # 2e7 # K

rho_gas = mu * mH * n

E_0 = 3.*kB*rho_gas*T / 2/mu/mH
E_previous = E_0

t = 0

tEnd = t_dyn_sec #1e7 * 365.25 * 24 * 3600

res = []

while t < tEnd:

	E_dot = dEdt(n, T, xi)
	E_new = E_previous + E_dot * dt
	
	E_previous = E_new
	
	T = 2.*mu*mH*E_new / 3./kB/rho_gas
	
	if T < 1.1e4:
		T = 1.1e4
	
	t += dt
	
	res.append([t/365.25/24/3600, T])
	


res = np.array(res)

t = res[:, 0]
T = res[:, 1]

print(np.sort(T))

plt.scatter(np.log10(t), T, s = 1, color = 'black')

plt.xlim(2, 9)
plt.ylim(1e4, 1e9)

plt.yscale('log')

plt.show()









