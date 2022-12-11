
import numpy as np
import matplotlib.pyplot as plt


def gammaH(J, T3): # It seems it should be T instead of T3. May be the original paper made a typo mistake !!!! Investigate later. 
	
	tmp1 = 1e-11 * T3**0.5 / (1. + 60./T3**4) + 1e-12 * T3
	tmp2 = 0.33 + 0.9 * np.exp(-1.*((J - 3.5)/0.9)**2)
	
	return tmp1 * tmp2


def gammaH2(J, T3):
	
	return (3.3e-12 + 6.6e-12 * T3) * (0.276 * J*J * np.exp(-(J/3.18)**1.7))



kB = 1.3807e-16  # cm2 g s-2 K-1

E0 = 0.0
E1 = 1.185e-14
E2 = 6.0 * E1
E3 = 12.0 * E1

E10 = 5860. * kB
E20 = 2. * E10


Tarr = np.logspace(1, 4, num = 100)

res = []

nHI = 1e4
nH2 = 1e4

n_tot = 1e4

for T in Tarr:

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
	
	#******* End of CO section *****************
	
	
	#******** Beginning of HII section **********
	
	if np.log10(T) <= 6.2:
		thetaT = 0.2 * (6.2 - np.log10(T))**4
	else:
		thetaT = 0.0
	
	L_HII = 1e-21 * (10**(-0.1-1.88*(5.23-np.log10(T))**4 + 10**(-1.7-thetaT)))
	
	#******* End of HII section ******************
	

	res.append([T, LH, LH2, L_CO])


res = np.array(res)

LH  = res[:, 1]
LH2 = res[:, 2]
LCO = res[:, 3]

#plt.plot(np.log10(Tarr), np.log10(LH), color = 'black')
plt.plot(np.log10(Tarr), np.log10(LH2), color = 'blue')
#plt.plot(np.log10(Tarr), np.log10(LCO), color = 'blue')
plt.xlim(1.5, 3.9)
plt.ylim(-33, -22)
#plt.ylim(-30, -21)

#plt.xscale('log')

plt.show()





