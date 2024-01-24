
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#===== Gam_e_H0
def Gam_e_H0(T):
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5)

#===== Gam_e_He0
def Gam_e_He0(T):
  return 2.38e-11 * T**0.5 * np.exp(-285335.4/T) / (1.0 + (T/1e5)**0.5)

#===== Gam_e_Hep
def Gam_e_Hep(T):
  return 5.68e-12 * T**0.5 * np.exp(-631515./T) / (1.0 + (T/1e5)**0.5)


#===== a_Hp
def a_Hp(T): 
  return 8.4e-11 / T**0.5 / (T/1e3)**0.2 / (1.0 + (T/1e6)**0.7)

#===== a_Hep
def a_Hep(T):
  return 1.5e-10 / T**0.6353

#===== a_d
def a_d(T):
  return 1.9e-3 / T**1.5 * np.exp(-470000./T) * (1.0 + 0.3 * np.exp(-94000./T))

#===== a_Hepp
def a_Hepp(T):
  return 3.36e-10 / T**0.5 / (T/1e3)**0.2 / (1. + (T/1e6)**0.7)



#-------------------------------------------------------
#-------------------- Cooling Rates --------------------
#-------------------------------------------------------
#-------------------------------------------------------
#-----------------------> He <--------------------------
#-------------------------------------------------------
#===== LambColExcH0 ---> erg.s^-1.cm^-3
def LambColExcH0(T, ne, nH0):
  return 7.50e-19 * np.exp(-118348/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LambColIonH0 ---> erg.s^-1.cm^-3
def LambColIonH0(T, ne, nH0):
  return 1.27e-21 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5) * ne * nH0

#===== LamRecHp ---> erg.s^-1.cm^-3
def LamRecHp(T, ne, nHp):
  return 8.70e-27 * T**0.5 / T**0.2 / (1.0 + (T/1e6)**0.7) * ne * nHp

#-------------------------------------------------------
#-----------------------> He <--------------------------
#-------------------------------------------------------
#===== LambColExcHep ---> erg.s^-1.cm^-3
def LambColExcHep(T, ne, nHep):
  return 5.54e-17 / T**0.397 * np.exp(-473638./T) / (1. + (T/1e5)**0.5) * ne * nHep

#===== LambColIonHe0 ---> erg.s^-1.cm^-3
def LambColIonHe0(T, ne, nHe0):
  return 9.38e-22 * T**0.5 * np.exp(-285335.4/T) / (1. + (T/1e5)**0.5) * ne * nHe0

#===== LambColIonHep ---> erg.s^-1.cm^-3
def LambColIonHep(T, ne, nHep):
  return 4.95e-22 * T**0.5 * np.exp(-631515./T) / (1. + (T/1e5)**0.5) * ne * nHep

#===== LambRecHep ---> erg.s^-1.cm^-3
def LambRecHep(T, ne, nHep):
  return 1.55e-26 * T**0.3647 * ne * nHep

#===== LambRecHepp ---> erg.s^-1.cm^-3
def LambRecHepp(T, ne, nHepp):
  return 3.48e-26 * T**0.5 / (T/1e3)**0.2 / (1. + (T/1e6)**0.7) * ne * nHepp

#===== LambRecHepp ---> erg.s^-1.cm^-3
def LambDiRecHep(T, ne, nHep):
  return 1.24e-13 / T**1.5 * np.exp(-470000./T) * (1. + 0.3 * np.exp(-94000./T)) * ne * nHep

#===== LambFree ---> erg.s^-1.cm^-3
def LambFree(T, nHp, nHep, nHepp, ne):
  gff = 1.1 + 0.34 * np.exp(-(5.5 - np.log10(T))**2/3.0)
  return 1.42e-27 * gff * T**0.5 * (nHp + nHep + 4. * nHepp) * ne




#===== Lambda
def Lambda(T, nH0, nHe0, nHep):
  nHp = nH - nH0
  nHepp = nH * y - (nHe0 + nHep)
  ne = (nH - nH0) + nHep + 2. * (nH * y - (nHe0 + nHep))
  return LambColExcH0(T, ne, nH0) + LambColIonH0(T, ne, nH0) + LamRecHp(T, ne, nHp) + \
         LambColExcHep(T, ne, nHep) + LambColIonHe0(T, ne, nHe0) + LambColIonHep(T, ne, nHep) + \
         LambRecHep(T, ne, nHep) + LambRecHepp(T, ne, nHepp) + LambDiRecHep(T, ne, nHep) + \
         LambFree(T, nHp, nHep, nHepp, ne)


#===== dSdt
def dSdt(t, S):
  y0, y1, y2, y3 = S

  nHp = nH - y1
  nHepp = nH * y - (y2 + y3)
  ne = (nH - y1) + y3 + 2. * (nH * y - (y2 + y3))

  Sum_of_n = y1 + nHp + y2 + y3 + nHepp + ne
  
  return [-1. * (gamma - 1.) / kB / Sum_of_n * Lambda(y0, y1, y2, y3),
          a_Hp(y0) * nHp * ne - Gam_e_H0(y0) * y1 * ne,
          (a_Hep(y0) + a_d(y0)) * y3 * ne - Gam_e_He0(y0) * y2 * ne,
          -Gam_e_Hep(y0) * y3 * ne - (a_Hep(y0) + a_d(y0)) * y3 * ne + \
          a_Hepp(y0) * nHepp * ne + Gam_e_He0(y0) * y2 * ne]





nH = 1000.
gamma = 5./3.
kB = 1.3807e-16

X = 0.76
Y = 0.24
y = Y / (4. - 4. * Y)

y0_0 = 1e6 # initial T
y1_0 = 0.001 # initial nH0
y2_0 = 0.001 # initial nHe0
y3_0 = 0.01 # initial nHep

S_0 = (y0_0, y1_0, y2_0, y3_0)

t = np.linspace(1, 3000, 1000) * 3.16e7

print(t)

sol = odeint(dSdt, y0 = S_0, t = t, tfirst = True)

T = sol[:, 0]
nH0 = sol[:, 1]
nHe0 = sol[:, 2]
nHep = sol[:, 3]

nHp = nH - nH0
nHepp = nH * y - (nHe0 + nHep)
ne = (nH - nH0) + nHep + 2. * (nH * y - (nHe0 + nHep))


print(sol)

t_yrs = t/3.16e7

plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.scatter(t_yrs, np.log10(T), s = 5, color = 'k')
#plt.xlim(0, max(t_yrs))
plt.ylim(3, 8)

plt.subplot(1, 2, 2)
#plt.scatter(t_yrs, nHp/nH, s = 5, color = 'k', label = 'nHp')
#plt.scatter(t_yrs, nH0/nH, s = 5, color = 'b', label = 'nH0')

plt.scatter(t_yrs, nHe0/nH, s = 5, color = 'orange', label = 'nHe0')
plt.scatter(t_yrs, nHep/nH, s = 5, color = 'lime', label = 'nHep')
plt.scatter(t_yrs, nHepp/nH, s = 5, color = 'yellow', label = 'nHepp')


#plt.scatter(t_yrs, ne/nH, s = 5, color = 'orange', label = 'ne')
plt.yscale('log')
plt.legend()

plt.savefig('myfig.png')

plt.show()




