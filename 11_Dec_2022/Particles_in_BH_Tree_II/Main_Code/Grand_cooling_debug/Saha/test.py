
import numpy as np
import matplotlib.pyplot as plt


#===== Lambda_e_H0
def Lambda_e_H0(T):
  T5 = T/1e5
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + T5**0.5)


#===== alpha_Hp
def alpha_Hp(T):
  T3 = T/1e3
  T6 = T/1e6
  return 8.4e-11 * T**-0.5 * T3**-0.2 / (1.0 + T6**0.7)


T = np.logspace(2, 9, 100)

L_e_H0 = np.vectorize(Lambda_e_H0)(T)
a_Hp = np.vectorize(alpha_Hp)(T)

n_H = 100. # cm^-3

n_Hp = 1.0 / (1.0 + a_Hp / L_e_H0) * n_H
n_H0 = n_H - n_Hp
n_e = n_Hp

hnu = 1.60218e-12 # erg ==> 13.6 eV

L = hnu * L_e_H0 * n_e * n_H0 / n_H/n_H

#plt.plot(np.log10(T), n_H0, color = 'k', label = 'n_H0')
#plt.plot(np.log10(T), n_Hp, color = 'b', label = 'n_Hp')
#plt.plot(np.log10(T), n_e, color = 'r', linestyle = '--', label = 'n_e')

plt.plot(np.log10(T), np.log10(L), color = 'k')

plt.ylim(-25, -22)

#plt.legend()

plt.show()

