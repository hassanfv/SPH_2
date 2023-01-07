
import numpy as np
from photolibs2 import *
import matplotlib.pyplot as plt

XH = 0.76
mH = 1.6726e-24 # gram

UnitTime_in_yrs = 500 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dt_t  = UnitTime_in_yrs * 3600. * 24. * 365.24

nHcgs = 0.1 # cm^-3   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rhot = nHcgs * mH / XH
print('nHcgs = ', nHcgs)

T = 100000. # K

uad = convert_Temp_to_u(T, nHcgs, XH)

print(f'uad = {uad:.3E}')

print('UnitTime_in_yrs = ', UnitTime_in_yrs)

ux = DoCooling_h(rhot, uad, dt_t, XH)

delta_u = uad - ux
print()
print('delta_u = ', delta_u/uad) # Normalized to better compare!

print()
print('u (before) = ', uad/uad)
print('u (After) = ', ux/uad)







