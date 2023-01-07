
import numpy as np
from photolibs3 import *
import matplotlib.pyplot as plt

XH = 0.76
mH = 1.6726e-24 # gram

UnitTime_in_yrs = 200 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dt_t  = UnitTime_in_yrs * 3600. * 24. * 365.24

nHcgs = 0.1 # cm^-3   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rhot = nHcgs * mH / XH
#print('nHcgs = ', nHcgs)

T = 100000. # K

uad = convert_Temp_to_u(T, nHcgs, XH)

#print(f'uad = {uad:.5E}')

print('UnitTime_in_yrs = ', UnitTime_in_yrs)


uad = 1.1884E+13
rho = 8.9824E-23
ux = 3.9120E+12

#uad, rhot, dtt, delta = [1.1884E+13,8.9824E-23,6.3114E+09,7.9720E+12]
uad, rhot, dtt, delta = [2.2042E+14, 2.3455E-25, 0.0, 0.0]

nHcgs = XH/mH * rhot
Temp = convert_u_to_temp_h(uad, nHcgs, XH)
print()
print('nHcgs = ', nHcgs)
print('Temp = ', Temp)

#uad = 1.0523E+13;
#rhot = 2.3518E-23;
#delta = 1.4857E+12;
uxx = uad - delta;

print()
print(f'U (BEFORE) = {uad:.5E}')
print(f'U (AFTER) = {uxx:.5E}')

ux = DoCooling_h(rhot, uad, dt_t, XH)

delta_u = uad - ux
#print()
#print('delta_u = ', delta_u/uad) # Normalized to better compare!

print()
print(f'u (before) = {uad:.4E}')
print(f'u (After) = {ux:.4E}')
print(f'uad - ux = {uad - ux:.4E}')
print()
print(f'u (After - NORMALIZED) = {ux/uad}')
print()
print(f'U AFTER / u After = {uxx/ux}')
print(f'u After / U AFTER = {ux/uxx}')







