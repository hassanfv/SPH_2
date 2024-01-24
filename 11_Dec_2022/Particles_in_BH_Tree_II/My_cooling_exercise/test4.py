
import numpy as np
import matplotlib.pyplot as plt


T = np.logspace(4, 8)


#===== Gam_e_H0
def Gam_e_H0(T):
  return 5.85e-11 * T**0.5 * np.exp(-157809.1/T) / (1.0 + (T/1e5)**0.5)

#===== a_Hp
def a_Hp(T): 
  return 8.4e-11 / T**0.5 / (T/1e3)**0.2 / (1.0 + (T/1e6)**0.7)

#===== func
def func(T, nH, nH0):
  
  return Gam_e_H0(T) * (nH - nH0) * nH0 - a_Hp(T) * (nH - nH0)**2


T = 20000
nH = 1.0

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


print(nH0_lower, nH0_upper)


nH0 = 0.5 * (nH0_lower + nH0_upper)

dnH0 = nH0

n_iter = 0

while (dnH0/nH0 > 1e-3) & (n_iter < 100):

  nH0_old = nH0
  if func(T, nH, nH0) > 0.0:
    nH0_upper = nH0
  else:
    nH0_lower = nH0
  
  nH0 = 0.5 * (nH0_lower + nH0_upper)
  dnH0 = np.abs(nH0_old - nH0)
  n_iter +=1
  
  

print()
print(f'At T = {T} K, we have nH0 = {nH0:.4f}')


