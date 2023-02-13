
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


#===== Mdot_in
def Mdot_in(Tou_in, L_AGN, clight, v_in):
	
	return Tou_in * L_AGN / clight / v_in


#===== P_b
def P_b(E_b, R_s):
	
	return E_b / 2.0 / np.pi / R_s**3


#===== P_0
def P_0(R_s, rho_0, R_0, alpha, R_gas, T_ism):

	return rho_0 * (R_s/R_0)**(-alpha) * R_gas * T_ism


#==== M_s
def M_s(R_s, rho_0, R_0, alpha):

	return 4.0 * np.pi * rho_0 * R_0**alpha * R_s**(3. - alpha) / (3. - alpha)


#===== M_t
def M_t(M_BH, sig, R_s, G):
	
	return M_BH + 2. * sig * sig * R_s / G


#===== rho_gas
def rho_gas(R_s, rho_0, R_0, alpha):

	return rho_0 * (R_s/R_0)**(-alpha)


#===== g_ff
def g_ff(Z_i, T):

	if T < 0:
		s()

	if T/Z_i/Z_i < 3.2e5:
	
		return 0.79464 + 0.1243 * np.log10(T/Z_i/Z_i)
	
	else:
	
		return 2.13164 - 0.124 * np.log10(T/Z_i/Z_i)


#===== M_sw; here we have t !
def M_sw(Tou_in, L_AGN, clight, v_in, t):

	return Mdot_in(Tou_in, L_AGN, clight, v_in) * t


#===== T_b; here we have t !
def T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t):

	Tb = 28./69. * E_b * mH / f_mix / M_sw(Tou_in, L_AGN, clight, v_in, t) / kB

	return Tb


#===== n_b_H; here we have t !
def n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t):

	return 3. * f_mix * M_sw(Tou_in, L_AGN, clight, v_in, t) * XH / 4.0 / np.pi / R_s**3 / mH


#===== Lambda_ff; here we have t !
def Lambda_ff(E_b, R_s, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t):

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t)
	nbe = 1.2 * nbH
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	
	Lambdaff = 1.426e-27 * Tb**0.5 * nbe * nbH * (g_ff(1., Tb) + 0.4 * g_ff(2., Tb))
	
	return Lambdaff


#===== Lambda_IC; here we have t !
def Lambda_IC(E_b, R_s, Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t):

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t)
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	
	Te = Beta * Tb
	ne = 1.22 * nbH
	
	if Te < 0:
		s()
	
	ln_Lambda = 39. + np.log(Te/1e10) - 0.5 * np.log(ne/1.)
	
	return 1.0e-19 * (Beta/0.1)**(-3./2.) * (Tb/1e8)**(-0.5) * (nbH/1.)**2 * (ln_Lambda/40.)


#===== L_b; here we have t !
def L_b(E_b, R_s, Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t):

	LambdaIC = Lambda_IC(E_b, R_s, Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t)
	
	Lambdaff = Lambda_ff(E_b, R_s, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t)

	
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	Te = Beta * Tb
	
	TC = 2e7 # K
	
	if (Te > 10.*TC) & (Te <= Tb):
		C = 5./3.
	else:
		C = 3.
	
	return 4./3. * np.pi * R_s**3 * (C * LambdaIC + Lambdaff)


#===== T_s
def T_s(E_s_th, R_s, rho_0, R_0, alpha):

	Ms = M_s(R_s, rho_0, R_0, alpha)
	Ts = 28./69. * E_s_th * mH / Ms / kB
	
	return Ts


#===== n_s_H
def n_s_H(E_b, E_s_th, R_s, rho_0, R_0, alpha):

	Ms = M_s(R_s, rho_0, R_0, alpha)
	Pb = P_b(E_b, R_s)

	return 3./2. * Pb * Ms * XH / mH / E_s_th


#===== L_s_ff
def L_s_ff(E_b, E_s_th, R_s, rho_0, R_0, alpha):

	Ms = M_s(R_s, rho_0, R_0, alpha)
	nsH = n_s_H(E_b, E_s_th, R_s, rho_0, R_0, alpha)
	nse = 1.20 * nsH
	Ts = T_s(E_s_th, R_s, rho_0, R_0, alpha)
	
	print((g_ff(1., Ts) + 0.4 * g_ff(4., Ts)))
	
	return 1.0/mH * 1.426e-27 * Ts**0.5 * nse * Ms * XH * (g_ff(1., Ts) + 0.4 * g_ff(4., Ts))


#===== L_s_Line
def L_s_Line(E_b, E_s_th, R_s, rho_0, R_0, alpha, Z, Z_sun, mH, XH):

	Ts = T_s(E_s_th, R_s, rho_0, R_0, alpha)
	
	if Ts <= 1e4:
	
		return 0.0
		
	else:
	
		Ms = M_s(R_s, rho_0, R_0, alpha)
		nsH = n_s_H(E_b, E_s_th, R_s, rho_0, R_0, alpha)
		nse = 1.20 * nsH
		
		if Ts <= 1e5:
			Beta_s = 2.0
		else:
			Beta_s = -0.7
		
		return 5.0e-22 * (Ts/1e5)**Beta_s * (Z/Z_sun) * (nse * Ms * XH / mH)


#===== L_s
def L_s(E_b, E_s_th, R_s, rho_0, R_0, alpha, Z, Z_sun, mH, XH):

	Lsff = L_s_ff(E_b, E_s_th, R_s, rho_0, R_0, alpha)
	Lsline = L_s_Line(E_b, E_s_th, R_s, rho_0, R_0, alpha, Z, Z_sun, mH, XH)
	
	return Lsff + Lsline




M_sun = 1.98892e33 # g

R_0 = 100 * 3.086e+18 # cm
sig = 200. * 1000. * 100 # cm/s
M_BH = 1e8 * M_sun
T_ism = 10000. # K
nH = 10
XH = 0.9
mH = 1.6726e-24
kB = 1.3807e-16
rho_0 = nH * mH / XH
G = 6.6738e-8
clight = 29979245800. # cm/s
#C = 5./3. # !!!!!!!!
f_mix = 1.0
Beta = 0.1
Tou_in = 1.0
alpha = 0.0 # uniform density!
L_AGN = 1e45 # erg/s
v_in = 10000 * 1000. * 100 # cm/s
R_gas = 8.31e7
Z = 1.0
Z_sun = 1.0

TFloor = 10000. # HFV: # To avoid negative E and T and hence NaN values - this E corresponds to T = 10000!


R_s_0 = 0.1 * 3.086e+18 # cm
v_s_0 = v_in
E_b_0 = 1.0e-10
E_s_th_0 = 1.0e-10

S_0 = (R_s_0, v_s_0, E_b_0, E_s_th_0) # initial condition vector !

Myrs = 1e6 * 365.25 * 24 * 3600

years_to_sec = 365.25 * 24. * 3600. # number of seconds in a year

Dt = 10.0 * years_to_sec # timeStep = 10 years !

tStart = 1.0 * years_to_sec
tEnd = 1e6 * years_to_sec

nSteps = int((tEnd - tStart) / Dt)

R_s = np.zeros(nSteps + 1)
v_s = np.zeros(nSteps + 1)
E_b = np.zeros(nSteps + 1)
E_s_th = np.zeros(nSteps + 1)

t = np.arange(tStart, tEnd, Dt)
tMyr = t / 1e6 / 365.25 / 24 / 3600

#==== Euler method ! =============
R_s[0] = R_s_0
v_s[0] = v_s_0
E_b[0] = E_b_0
E_s_th[0] = E_s_th_0

for i in range(1, nSteps):

	Ms = M_s(R_s[i-1], rho_0, R_0, alpha)
	Pb = P_b(E_b[i-1], R_s[i-1])
	P0 = P_0(R_s[i-1], rho_0, R_0, alpha, R_gas, T_ism)
	Mt = M_t(M_BH, sig, R_s[i-1], G)
	rhoGas = rho_gas(R_s[i-1], rho_0, R_0, alpha)
	

	v_dot_s = 1./Ms * (4.*np.pi*R_s[i-1]**2 * (Pb - P0) - G*Ms*Mt/R_s[i-1]**2 - 4.*np.pi*rhoGas*R_s[i-1]**2 * v_s[i-1]**2)

	MdotIn = Mdot_in(Tou_in, L_AGN, clight, v_in)
	Lb = L_b(E_b[i-1], R_s[i-1], Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t[i])
	

	E_dot_b = 0.5 * MdotIn * v_in**2 - 4.*np.pi*R_s[i-1]**2 * (Pb - P0) * v_s[i-1] - Lb


	#-----
	Ls = L_s(E_b[i-1], E_s_th[i-1], R_s[i-1], rho_0, R_0, alpha, Z, Z_sun, mH, XH)
	E_dot_s_tot = 4. * np.pi * R_s[i-1]**2 * (Pb - P0) * v_s[i-1] - G * Ms * Mt / R_s[i-1]**2 * v_s[i-1] - Ls
	E_dot_s_kin = Ms * v_s[i-1] * v_dot_s + 2.0 * np.pi * R_s[i-1]**2 * rhoGas * v_s[i-1]**3
	E_dot_s_th = E_dot_s_tot - E_dot_s_kin

	
	R_s[i] = R_s[i-1] + Dt * v_s[i]
	v_s[i] = v_s[i-1] + Dt * v_dot_s
	E_b[i] = E_b[i-1] + Dt * E_dot_b
	E_s_th[i] = E_s_th[i-1] + Dt * E_dot_s_th


print(E_s_th)

s()


Rs_kpc = R_s / 3.086e+18 / 1000
vs_kms = v_s / 100 / 1000

Ts = T_s(E_s_th, R_s, rho_0, R_0, alpha)

#Ts = [1e4 if x < 1e4 else x for x in Ts]

if True:
	plt.scatter(Rs_kpc, vs_kms, s = 1)
	plt.yscale('log')
	plt.xscale('log')
	plt.ylim(1e2, 1e4)
	plt.xlim(2e-2, 2)
	plt.show()



