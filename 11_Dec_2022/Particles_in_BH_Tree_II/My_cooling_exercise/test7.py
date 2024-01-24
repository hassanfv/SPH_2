
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

  while (dnH0/nH0 > 1e-6) & (n_iter < 100):

    nH0_old = nH0
    if func(T, nH, nH0) > 0.0:
      nH0_upper = nH0
    else:
      nH0_lower = nH0
    
    nH0 = 0.5 * (nH0_lower + nH0_upper)
    dnH0 = np.abs(nH0_old - nH0)
    
    n_iter +=1

  return nH0



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

#===== getCooling
def getCooling(Temp, nH):
  nH0 = findAbund(Temp, nH)
  nHp = nH - nH0
  ne = nHp
  return (LambColExcH0(Temp, ne, nH0) + LambColIonH0(Temp, ne, nH0) + LamRecHp(Temp, ne, nHp)) / nH / nH


#===== rootFunc
def rootFunc(T, T0, nH, dt):

  gamma = 5./3.
  kB = 1.3807e-16  # cm2 g s-2 K-1
  
  Lamb = getCooling(T, nH) * nH * nH
  nH0 = findAbund(T, nH)
  nHp = nH - nH0
  ne = nHp
  n = nH0 + nHp + ne

  return T - (T0 - (gamma - 1.) / kB / n * Lamb * dt)
  
  


# T0 is the initial T

#===== doCooling
def doCooling(T0, nH, dt):

  T = T0

  if rootFunc(T, T0, nH, dt) > 0.0:
    T_upper = T0
    T_lower = T_upper / 1.1
    while (rootFunc(T_lower, T0, nH, dt) > 0.0):
      T_lower /= 1.1
    T_lower = T_lower
    T_upper = T_lower * 1.1
    
  if rootFunc(T, T0, nH, dt) < 0.0:
    T_lower = T0
    T_upper = T_lower * 1.1
    while(rootFunc(T_upper, T0, nH, dt) < 0.0):
      T_upper *= 1.1
    
    T_upper = T_upper
    T_lower = T_upper / 1.1


  T = 0.5 * (T_upper + T_lower)
  dT = T
  n_iter = 0

  while (dT/T > 1e-6):
    
    T_old = T
    
    if rootFunc(T, T0, nH, dt) > 0.0:
      T_upper = T
    else:
      T_lower = T
    
    T = 0.5 * (T_upper + T_lower)
    dT = abs(T - T_old)
    n_iter += 1

    if n_iter > 100:
      print('MAX n_iter reached !!!!')
      
  return T


T0 = 1e6
nH = 100.0
dt = 3.16e7

N = 1000 # number of time steps to evolve the gas temperature

t = 0
res = []

for i in range(N):

  T = doCooling(T0, nH, dt)
  T0 = T
  t += dt
  
  res.append([t, T])

res = np.array(res)

t_yrs = res[:, 0] / 3600 / 24 / 365.25
T = res[:, 1]

plt.scatter(t_yrs, np.log10(T), s = 5, color = 'k')

plt.ylim(4, 8)

plt.show()




