
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

	Mol_ism = 1.3 # g/mol
	R_specific = R_gas / Mol_ism

	return rho_0 * (R_s/R_0)**(-alpha) * R_specific * T_ism


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
	ne = 1.22 * nbH
	
	#ln_Lambda = 39. + np.log(Te/1e10) - 0.5 * np.log(ne/1.)
	
	return 1.0e-19 * (Beta/0.1)**(-3./2.) * (Tb/1e8)**(-0.5) * (nbH/1.)**2 * (ln_Lambda/40.)




#===== t_p_cool (FGQ12)
def t_p_cool(R_s, v_s, E_b, tt):

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, tt)
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, tt)
	
	Te = Beta * Tb
	ne = 1.22 * nbH
	
	ln_Lambda = 39. + np.log(Te/1e10) - 0.5 * np.log(ne/1.)
	
	R_s_kpc = R_s / 3.086e+18 / 1000.
	v_s_kms = v_s / 100. / 1000.
	v_in_kms = v_in / 100. / 1000.
	tpc_in_yr = 1.49e9 * R_s_kpc**2 * (L_AGN/1e46)**(-1) * (v_s_kms/1000.)**(2./5.) * (v_in_kms / 30000)**(8./5.) * (ln_Lambda/40.)**(-2/5) * Tou_in**(-2./5.)
	
	tpc_in_sec = tpc_in_yr * 365.25 * 24. * 3600.
	
	return tpc_in_sec





#===== t_ff (FGQ12)
def t_ff(R_s, E_b, t):

	gB = 1.
	C = 1.

	nbH = n_b_H(R_s, f_mix, mH, XH, Tou_in, L_AGN, clight, v_in, t)
	Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t)
	
	Te = Beta * Tb
	ne = 1.22 * nbH
	
	tff_in_yr = 4.7e6 / gB / C * (Te / 1e6)**0.5 /ne
	tff_in_sec = tff_in_yr * 365.25 * 24. * 3600.
	
	return tff_in_sec



#===== t_R_Comp (FGQ12)
def t_R_Comp(R_s, v_s):

	R_s_kpc = R_s / 3.086e+18 / 1000.
	v_s_kms = v_s / 100. / 1000.
	v_in_kms = v_in / 100. / 1000.
	
	tRComp_in_yr = 2e7 * (R_s_kpc)**2 * (L_AGN / 1e46)**(-1.) * (v_in_kms/30000.)**(-2.)
	tRComp_in_sec = tRComp_in_yr * 365.25 * 24. * 3600.
	
	return tRComp_in_sec



#===== L_b; here we have t !
def L_b(E_b, R_s, v_s, t):

	tpcool = t_p_cool(R_s, v_s, E_b)
	tff = t_ff(R_s, E_b, t)

	LambdaIC = mu * E_b / tpcool
	Lambdaff = mu * E_b / tff

	Lb = LambdaIC + Lambdaff
	
	return Lb



def dSdt(tt, S):
	
	R_s, v_s, E_b = S

	Ms = M_s(R_s, rho_0, R_0, alpha)
	Pb = P_b(E_b, R_s)
	P0 = P_0(R_s, rho_0, R_0, alpha, R_gas, T_ism)
	Mt = M_t(M_BH, sig, R_s, G)
	rhoGas = rho_gas(R_s, rho_0, R_0, alpha)
	
	v_dot_s = 1./Ms * (4.*np.pi*R_s**2 * (Pb - P0) - G*Ms*Mt/R_s**2 - 4.*np.pi*rhoGas*R_s**2 * v_s**2)
	
	
	MdotIn = Mdot_in(Tou_in, L_AGN, clight, v_in)
	#---- L_b ------------
	#Lb = L_b(E_b, R_s, v_s, tt)
	tpcool = t_p_cool(R_s, v_s, E_b, tt)
	tff = t_ff(R_s, E_b, tt)

	LambdaIC = mu * E_b / tpcool
	Lambdaff = mu * E_b / tff

	Lb = LambdaIC + Lambdaff
	#---------------------
	
	#print(f'{E_b:.2E}, {tpcool / 365.25 / 24 / 3600:.2E}, {tff / 365.25 / 24 / 3600:.2E}')
	
	#E_dot_b = 0.5 * MdotIn * v_in**2 - 4.*np.pi*R_s**2 * (Pb - P0) * v_s - Lb
	E_dot_b = 0.5 * MdotIn * v_in**2 - 4.*np.pi*R_s**2 * Pb * v_s - Lb
	
	return [v_s, v_dot_s, E_dot_b]




M_sun = 1.98892e33 # g

#ln_Lambda = 40.
mu = 0.5
R_0 = 100 * 3.086e+18 # cm
sig = 200. * 1000. * 100 # cm/s
M_BH = 1e8 * M_sun
T_ism = 10000. # K
nH = 550.     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
alpha = 1.0 # uniform density!
L_AGN = 1e46 # erg/s  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
v_in = 30000 * 1000. * 100 # cm/s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
R_gas = 8.31e7
Z = 1.0   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Z_sun = 1.0


R_s_0 = 0.1 * 3.086e+18 # cm
v_s_0 = v_in
E_b_0 = 1.0e-10

S_0 = (R_s_0, v_s_0, E_b_0) # initial condition vector !

oneMyrs = 1.0e7* 365.25 * 24 * 3600

t = np.linspace(10, oneMyrs, 1000000)

res = odeint(dSdt, y0 = S_0, t = t, tfirst = True, rtol=1e-2, atol=1e-2)
print(res)

R_s = res[:, 0]
v_s = res[:, 1]
E_b = res[:, 2]


Rs_kpc = R_s / 3.086e+18 / 1000
vs_kms = v_s / 100 / 1000

t = t[1:]
tMyr = t / 1e6 / 365.25 / 24 / 3600
vs_kms = vs_kms[1:]
Rs_kpc = Rs_kpc[1:]
E_b = E_b[1:]

#---- Converting E_b to T_b
Tb = T_b(E_b, mH, kB, f_mix, Tou_in, L_AGN, clight, v_in, t) # Note that t is also an array !
#------

tyr = t / 3600 / 24 / 365.25

if True:
	plt.plot(np.log10(tyr), np.log10(Tb), color = 'k', linewidth = 3)
	#plt.plot(tMyr, np.log10(vs_kms), color = 'k', linewidth = 3)
	#plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(2, max(np.log10(tyr)))
	#plt.ylim(-1, 5)
	plt.show()



if False:
	plt.plot(Rs_kpc, vs_kms, color = 'k', linewidth = 3)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(1e-4, 10)
	plt.ylim(1e2, 4e4)
	plt.show()





