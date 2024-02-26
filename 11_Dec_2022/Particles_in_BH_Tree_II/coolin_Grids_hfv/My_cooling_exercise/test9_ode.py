
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#===== Gam_e_H0
def Gam_e_H0(T):
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5)

#===== a_Hp
def a_Hp(T): 
  return 8.4e-11 / T**0.5 / (T/1e3)**0.2 / (1.0 + (T/1e6)**0.7)


#===== LambColExcH0 ---> erg.s^-1.cm^-3
def LambColExcH0(T, ne, nH0):
  return 7.50e-19 * np.exp(-118348/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LambColIonH0 ---> erg.s^-1.cm^-3
def LambColIonH0(T, ne, nH0):
  return 1.27e-21 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LamRecHp ---> erg.s^-1.cm^-3
def LamRecHp(T, ne, nHp):
  return 8.70e-27 * T**0.5 / T**0.2 / (1.0 + (T/1e6)**0.7) * ne * nHp


#===== Lambda
def Lambda(T, nH0):
  nHp = nH - nH0
  ne = nHp
  return LambColExcH0(T, ne, nH0) + LambColIonH0(T, ne, nH0) + LamRecHp(T, ne, nHp)


#===== dSdt
def dSdt(t, S):
  y1, y2 = S
  
  return [-1. * (gamma - 1.) / kB / (2.*nH + y2) * Lambda(y1, y2),
          a_Hp(y1) * (nH - y2)**2 - Gam_e_H0(y1) * y2 * (nH - y2)]

nH = 100.
gamma = 5./3.
kB = 1.3807e-16
y1_0 = 1e6 # initial T
y2_0 = 0.0002 # initial nH0

S_0 = (y1_0, y2_0)

t = np.linspace(1, 120000, 10000) * 3.16e7

print(t)

sol = odeint(dSdt, y0 = S_0, t = t, tfirst = True)

T = sol[:, 0]
nH0 = sol[:, 1]

nHp = nH - nH0
ne = nHp

print(sol)

t_yrs = t/3.16e7

plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.scatter(t_yrs, np.log10(T), s = 5, color = 'k')
#plt.xlim(0, max(t_yrs))
plt.ylim(3, 8)

plt.subplot(1, 2, 2)
plt.scatter(t_yrs, nHp, s = 5, color = 'k', label = 'nHp')
plt.scatter(t_yrs, nH0, s = 5, color = 'b', label = 'nH0')
plt.scatter(t_yrs, ne, s = 5, color = 'orange', label = 'ne')
plt.yscale('log')
plt.legend()

plt.savefig('myOnlyH.png')

plt.show()




