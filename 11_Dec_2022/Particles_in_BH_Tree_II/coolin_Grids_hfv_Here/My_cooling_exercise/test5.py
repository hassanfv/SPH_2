
import numpy as np
import matplotlib.pyplot as plt


#===== Gam_e_H0
def Gam_e_H0(T):
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5)

#===== a_Hp
def a_Hp(T): 
  return 8.4e-11 / T**0.5 / (T/1e3)**0.2 / (1.0 + (T/1e6)**0.7)

#===== func
def func(T, nH, nH0):
  return Gam_e_H0(T) * (nH - nH0) * nH0 - a_Hp(T) * (nH - nH0)**2


#===== findAbund
def findAbund(T, nH):

  nH0 = 0.5 * nH # initial guess!

  if func(T, nH, nH0) > 0.0:
    nH0_upper = nH0
    nH0_lower = nH0_upper / 1.1
    
    while func(T, nH, nH0_lower) > 0.0:
      nH0_lower /= 1.1
    
    nH0_lower = nH0_lower
    nH0_upper = nH0_lower * 1.1

  else:
    nH0_lower = nH0
    nH0_upper = nH0_lower * 1.1
    
    while func(T, nH, nH0_upper) < 0.0:
      nH0_upper *= 1.1
    
    nH0_upper = nH0_upper
    nH0_lower /= nH0_upper


  nH0 = 0.5 * (nH0_lower + nH0_upper)

  dnH0 = nH0

  n_iter = 0

  while (dnH0/nH0 > 1e-3) & (n_iter < 100):

    if func(T, nH, nH0) > 0.0:
      nH0_upper = nH0
    else:
      nH0_lower = nH0
    
    dnH0 = np.abs(nH0_lower - nH0_upper)
    nH0 = 0.5 * (nH0_lower + nH0_upper)
    n_iter +=1

  return nH0


T = 20000
nH = 100.0

nH0 = findAbund(T, nH)
print()
print(f'At T = {T} K, we have nH0 = {nH0:.4f}')


#===== LambColExcH0 ---> erg.s^-1.cm^-3
def LambColExcH0(T, ne, nH0):
  return 7.50e-19 * np.exp(-118348/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LambColIonH0 ---> erg.s^-1.cm^-3
def LambColIonH0(T, ne, nH0):
  return 1.27e-21 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LamRecHp ---> erg.s^-1.cm^-3
def LamRecHp(T, ne, nHp):
  return 8.70e-27 * T**0.5 / T**0.2 / (1.0 + (T/1e6)**0.7) * ne * nHp


T = np.logspace(4.1, 8, 1000)
nH = 100.0

Lamb = []

for Temp in T:

  nH0 = findAbund(Temp, nH)
  nHp = nH - nH0
  ne = nHp

  Lamb.append([LambColExcH0(Temp, ne, nH0), LambColIonH0(Temp, ne, nH0), LamRecHp(Temp, ne, nHp)])


Lamb = np.array(Lamb) / nH/nH

print(Lamb.shape)


L_ColExcH0 = np.log10(Lamb[:, 0])
L_ColIonH0 = np.log10(Lamb[:, 1])
L_RecHp = np.log10(Lamb[:, 2])

L_total = np.log10(Lamb[:, 0] + Lamb[:, 1] + Lamb[:, 2])

plt.scatter(np.log10(T), L_ColExcH0, s = 5, color = 'orange', label = 'L_ColExcH0')
plt.scatter(np.log10(T), L_ColIonH0, s = 5, color = 'lime', label = 'L_ColIonH0')
plt.scatter(np.log10(T), L_RecHp, s = 5, color = 'blue', label = 'L_RecHp')

plt.scatter(np.log10(T), L_total, s = 5, color = 'black', label = 'Total')

plt.ylim(-25, -21.5)

plt.legend()

plt.savefig('Lamb.png')

plt.show()





