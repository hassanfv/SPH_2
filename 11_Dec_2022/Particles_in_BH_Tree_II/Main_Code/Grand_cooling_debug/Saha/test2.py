
import numpy as np
import matplotlib.pyplot as plt

hnu_H = 2.17e-11 # erg ==> 13.6 eV

#===== Lambda_e_H0
def Lambda_e_H0(T):
  T5 = T/1e5
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + T5**0.5)

#===== alpha_Hp
def alpha_Hp(T):
  T3 = T/1e3
  T6 = T/1e6
  return 8.4e-11 * T**(-0.5) * T3**(-0.2) / (1.0 + T6**(0.7)) # Katz et al - Table_2
  #return 8.7e-27 * T**0.5 * T3**-0.2 / (1.0 + T6**0.7) #/  hnu_H #!!!!!!!!!!!!!!!!!!!!!!!!! Katz et al - Table_1

#===== Lambda_e_He0
def Lambda_e_He0(T):
  T5 = T/1e5
  return 2.38e-11 * T**0.5 * np.exp(-285335.4/T) / (1.0 + T5**0.5)

#===== Lambda_e_Hep
def Lambda_e_Hep(T):
  T5 = T/1e5
  return 5.68e-12 * T**0.5 * np.exp(-631515.0/T) / (1.0 + T5**0.5)

#===== alpha_Hep
def alpha_Hep(T):
  return 1.5e-10 * T**-0.6353

#===== alpha_d
def alpha_d(T):
  return 1.9e-3 * T**-1.5 * np.exp(-470000.0/T) * (1.0 + 0.3 * np.exp(-94000.0/T))

def alpha_Hepp(T):
  T3 = T / 1e3
  T6 = T / 1e6
  return 3.36e-10 * T**-0.5 * T3**-0.2 / (1.0 + T6**0.7)



T = np.logspace(2, 9, 100)

L_e_H0 = np.vectorize(Lambda_e_H0)(T)
a_Hp = np.vectorize(alpha_Hp)(T)

L_e_He0 = np.vectorize(Lambda_e_He0)(T)
L_e_Hep = np.vectorize(Lambda_e_Hep)(T)

a_Hep = np.vectorize(alpha_Hep)(T)
a_d = np.vectorize(alpha_d)(T)
a_Hepp = np.vectorize(alpha_Hepp)(T)


n_H = 1. # cm^-3

Y = 0.24
y = Y / (4.0 - 4.0 * Y)

n_H0 = n_H * a_Hp / (a_Hp + L_e_H0)
n_Hp = n_H - n_H0
n_Hep = y * n_H / (1.0 + (a_Hep + a_d)/L_e_He0 + L_e_Hep / a_Hepp)
n_He0 = n_Hep * (a_Hep + a_d)/L_e_He0
n_Hepp = n_Hep * L_e_Hep / a_Hepp
n_e = n_Hp + n_Hep + 2. * n_Hepp

#for i in range(len(T)):
#  print(f'T = {T[i]}, n_H0 = {n_H0[i]},  n_Hp = {n_Hp[i]},  n_Hep = {n_Hep[i]},  n_He0 = {n_He0[i]},  n_Hepp = {n_Hepp[i]},  n_e = {n_e[i]}')
#  print()

print()
print(n_H0)

recRate_H = 1.036e-16 * T * a_Hp * n_e * n_Hp / n_H / n_H # recommbination rate # Note we dont have hnu_H as we use Katz Table 1

ionRate_H = L_e_H0 * n_e * n_H0 * hnu_H / n_H / n_H # ionization rate

print()
#print(recRate_H)

plt.plot(np.log10(T), np.log10(recRate_H), color = 'r')
plt.plot(np.log10(T), np.log10(ionRate_H), color = 'b', linestyle = '--')

plt.xlim(2, 9)
plt.ylim(-25, -21)

plt.show()


