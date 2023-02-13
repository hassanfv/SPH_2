
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


	if T/Z_i/Z_i < 3.2e5:
	
		return 0.79464 + 0.1243 * np.log10(T/Z_i/Z_i)
	
	else:
	
		return 2.13164 - 0.124 * np.log10(T/Z_i/Z_i)


#===== M_sw; here we have t !
def M_sw(Tou_in, L_AGN, clight, v_in, t):

	return Mdot_in(Tou_in, L_AGN, clight, v_in) * t


#===== T_b; here we have t !
def T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t):

	return 28./69. * E_b * mH / f_mix / M_sw(Tou_in, L_AGN, clight, v_in, t) / kB


#===== n_b_H; here we have t !
def n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t):

	return 3. * f_mix * M_sw(Tou_in, L_AGN, clight, v_in, t) * XH / 4.0 / np.pi / R_s**3 / mH


#===== Lambda_ff; here we have t !
def Lambda_ff(E_b, R_s, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t):

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t)
	nbe = 1.2 * nbH
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	return 1.426e-27 * Tb**0.5 * nbe * nbH * (g_ff(1., Tb) + 0.4 * g_ff(2., Tb))


#===== Lambda_IC; here we have t !
def Lambda_IC(E_b, R_s, Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t):

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t)
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	
	Te = Beta * Tb
	ne = 1.2 * nbH
	
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



def dSdt(t, S):

	R_s, v_s, E_b = S

	
	Ms = M_s(R_s, rho_0, R_0, alpha)
	Pb = P_b(E_b, R_s)
	P0 = P_0(R_s, rho_0, R_0, alpha, R_gas, T_ism)
	Mt = M_t(M_BH, sig, R_s, G)
	rhoGas = rho_gas(R_s, rho_0, R_0, alpha)
	
	v_dot_s = 1./Ms * (4.*np.pi*R_s**2 * (Pb - P0) - G*Ms*Mt/R_s**2 - 4.*np.pi*rhoGas*R_s**2 * v_s**2)
	
	
	MdotIn = Mdot_in(Tou_in, L_AGN, clight, v_in)
	Lb = L_b(E_b, R_s, Beta, f_mix, mH, kB, XH, Tou_in, L_AGN, clight, v_in, t)
	
	E_dot_b = 0.5 * MdotIn * v_in**2 - 4.*np.pi*R_s**2 * (Pb - P0) * v_s - Lb
	
	
	#if v_s > v_in:
	#	v_s = v_in
	#	v_dot_s = -87.63387875119089
	
	return [v_s,
		v_dot_s,
		E_dot_b]


M_sun = 1.98892e33 # g

R_0 = 100 * 3.086e+18 # cm
sig = 200. * 1000. * 100 # cm/s
M_BH = 1e8 * M_sun
T_ism = 10000. # K
nH = 1.0e1
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
v_in = 30000 * 1000. * 100 # cm/s
R_gas = 8.31e7

R_s_0 = 0.1 * 3.086e+18 # cm
v_s_0 = v_in
E_b_0 = 1e-10

S_0 = (R_s_0, v_s_0, E_b_0) # initial condition vector !

Myrs = 1e6 * 365.25 * 24 * 3600

t = np.logspace(np.log10(10), np.log10(Myrs), 1000)
tMyr = t / 1e6 / 365.25 / 24 / 3600

print(max(t))

res = odeint(dSdt, y0 = S_0, t = t, tfirst = True, rtol=1e-1, atol=1e-1)
#res = solve_ivp(dSdt, (10, 31557600000000.0), y0 = [R_s_0, v_s_0, E_b_0], t_eval = t, method='DOP853', rtol=1e-10, atol=1e-13)

R_s = res[:, 0]
v_s = res[:, 1]
E_b = res[:, 2]

Rs_kpc = R_s / 3.086e+18 / 1000
vs_kms = v_s / 100 / 1000

print(res)

plt.scatter(tMyr, vs_kms, s = 1)

plt.xscale('log')
plt.yscale('log')

#plt.xlim(2e-2, 2)
plt.ylim(1e2, 1e5)

plt.show()



