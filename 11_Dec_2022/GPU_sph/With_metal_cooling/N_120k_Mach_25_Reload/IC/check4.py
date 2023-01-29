
import numpy as np
from photolibs import *
import matplotlib.pyplot as plt


#---------- Units for Infall - Outfall ----------
Mref = 367.791 * 1.989e33 # gram # # calculated using the nH_M_R_estimator.py code here !
Rref = 30.0*3.086e18 # cm (this is 10 pc)
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

dt = 1e-3

rhot = 0.014 # in code unit
ut = 4000.0  # in code unit

ut = ut*Unit_u_in_cgs
rhot = rhot * UnitDensity_in_cgs
nHcgs    = XH * rhot / mH

print('nHcgs = ', nHcgs)

print()
print(f'u_cgs = {ut:.3E}')
print()

T = convert_u_to_temp_h(ut, nHcgs, XH)

print('T = ', T)


#cr = coolingRateFromU_h(ut, nHcgs, XH)

dt_t  = dt * UnitTime_in_s

ux = DoCooling_h(rhot, ut, dt_t, XH)

print()
print('u (before) = ', ut/Unit_u_in_cgs)
print('u (After) = ', ux/Unit_u_in_cgs)







