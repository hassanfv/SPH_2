
import numpy as np
from photolibs2 import *
import matplotlib.pyplot as plt


#---------- Units for Infall - Outfall ----------
Mref = 10000.0 * 1.989e33 # gram (this is 10000 M_sun)
Rref = 10.0*3.086e18 # cm (this is 10 pc)
grav_const = 6.67259e-8 #  cm3 g-1 s-2

UnitDensity_in_cgs = 3. * Mref / 4. / np.pi / Rref**3
Unit_u_in_cgs = grav_const * Mref / Rref
UnitTime_in_s = np.sqrt(Rref**3 / grav_const / Mref)
UnitTime_in_yrs = UnitTime_in_s / 3600.0 / 24.0 / 365.25
UnitVelocity_in_cm_per_s = np.sqrt(grav_const * Mref / Rref)
UnitVelocity_in_km_per_s = UnitVelocity_in_cm_per_s / 100.0 / 1000.0
#------------------------------------------------

print('Unit_u_in_cgs = ', Unit_u_in_cgs)

XH = 0.76
mH = 1.6726e-24 # gram

dt = 1e-4

rhot = 0.001 # in code unit
ut = 60.0  # in code unit

ut = ut*Unit_u_in_cgs
rhot = rhot * UnitDensity_in_cgs
nHcgs    = XH * rhot / mH

print('nHcgs = ', nHcgs)

T = convert_u_to_temp_h(ut, nHcgs, XH)
print('T = ', T)
#u = convert_Temp_to_u(T, nHcgs, XH)


Tmin = 1e4
Tmax = 1e6
Tgrid = np.logspace(np.log10(Tmin), np.log10(Tmax), 1000)
# converting T to u
#uGrid = np.array([convert_Temp_to_u(T, nHcgs, XH) for T in Tgrid])

#print(uGrid)

#cr = coolingRateFromU_h(ut, nHcgs, XH)

dt_t  = dt * UnitTime_in_s

ux = DoCooling_h(rhot, ut, dt_t, XH)

print()
print('u (before) = ', ut/Unit_u_in_cgs)
print('u (After) = ', ux/Unit_u_in_cgs)







