
import numpy as np
import matplotlib.pyplot as plt


def gammaH(J, T3):
	
	tmp1 = 1e-11 * T3**0.5 / (1. + 60./T3**4) + 1e-12 * T3
	tmp2 = 0.33 + 0.9 * np.exp(-1.*((J - 3.5)/0.9)**2)
	
	return tmp1 * tmp2


kB = 1.3807e-16  # cm2 g s-2 K-1

E0 = 0.0
E1 = 1.185e-14
E2 = 6.0 * E1
E3 = 12.0 * E1

E10 = 5860. * kB
E20 = 2. * E10


Tarr = np.logspace(1, 4, num = 100)

res = []

nHI = 1e8

for T in Tarr:

	T3 = T / 1000.0

	#-------------- L_r_H -----------------
	L_r_H_1 = (9.5e-22 *T3**3.76/(1.+0.12*T3**2.1)) * np.exp(-1.*(0.13/T3)**3)
	L_r_H_2 = 3e-24 * np.exp(-0.51/T3)
	L_r_H = 1./nHI * (L_r_H_1 + L_r_H_2)


	L_r_H_n_0_1 = 5. * gammaH(2, T3) * np.exp(-1.*(E2 - E0)/kB/T) * (E2 - E0)
	L_r_H_n_0_2 = 7./3. * gammaH(3, T3) * np.exp(-1.*(E3 - E1)/kB/T) * (E3 - E1)
	L_r_H_n_0 = 0.25*L_r_H_n_0_1 + 0.75*L_r_H_n_0_2

	nHrot_nHI = L_r_H / L_r_H_n_0

	H_rot = L_r_H / (1. + nHrot_nHI)
	#--------------------------------------


	L_v_H = 1./nHI * (6.7e-19 * np.exp(-5.86/T3) + 1.6e-18 * np.exp(-11.7/T3))
	L_v_H_n_0 = gammaH(10, T3) * np.exp(-E10/kB/T) * E10 + gammaH(20, T) * np.exp(-E20/kB/T) * E20

	nHvib_nHI = L_v_H / L_v_H_n_0

	H_vib = L_v_H / (1. + nHvib_nHI)

	LH = H_rot + H_vib

	res.append([T, LH])


res = np.array(res)

LH = res[:, 1]
logT = res[:, 0]

plt.plot(np.log10(Tarr), np.log10(LH))
plt.xlim(1.5, 3.9)
plt.ylim(-33, -22)

#plt.xscale('log')

plt.show()





