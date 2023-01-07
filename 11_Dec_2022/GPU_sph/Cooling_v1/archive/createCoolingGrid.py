
import numpy as np
from photolibs2 import *
import time
import pickle


#---------- Units for Infall - Outfall ----------
Mref = 100.0 * 1.989e33 # gram (this is 10000 M_sun)
Rref = 10.0*3.086e18 # cm (this is 10 pc)
grav_const = 6.67259e-8 #  cm3 g-1 s-2

UnitDensity_in_cgs = 3. * Mref / 4. / np.pi / Rref**3
Unit_u_in_cgs = grav_const * Mref / Rref
UnitTime_in_s = np.sqrt(Rref**3 / grav_const / Mref)
UnitTime_in_yrs = UnitTime_in_s / 3600.0 / 24.0 / 365.25
UnitVelocity_in_cm_per_s = np.sqrt(grav_const * Mref / Rref)
UnitVelocity_in_km_per_s = UnitVelocity_in_cm_per_s / 100.0 / 1000.0
#------------------------------------------------

print('UnitTime_in_yrs = ', UnitTime_in_yrs)
print('Unit_u_in_cgs = ', Unit_u_in_cgs)

XH = 0.76
mH = 1.6726e-24 # gram

dt = 1e-4

rhot = 10. # in code unit
ut = 60.0  # in code unit

ut = ut*Unit_u_in_cgs
rhot = rhot * UnitDensity_in_cgs
nHcgs    = XH * rhot / mH

print('nHcgs = ', nHcgs)

#T = convert_u_to_temp_h(ut, nHcgs, XH)
#print('T = ', T)
#u = convert_Temp_to_u(T, nHcgs, XH)

N_u = 10
N_rho = 100

Tmin = 1e4
Tmax = 1e6
Tgrid = np.logspace(np.log10(Tmin), np.log10(Tmax), N_u)
# converting T to u
uGrid = np.array([convert_Temp_to_u(T, nHcgs, XH) for T in Tgrid])
#print(uGrid)

nH_min = 1e-4
nH_max = 1e3
rho_min = nH_min * mH
rho_max = nH_max * mH
rhoGrid = np.logspace(np.log10(rho_min), np.log10(rho_max), N_rho)

dt_t  = 500 * 3600. * 24. * 365.24 # 500 YEARS

N_u = len(uGrid)
N_rho = len(rhoGrid)
res = np.zeros((N_u*N_rho, 4))

print(res.shape)
print(res.nbytes/1024/1024) # in Mb

k = 0

TA = time.time()

# Each loop takes 0.133 seconds!

for i in range(len(rhoGrid)):

	for j in range(len(uGrid)):
		
		ux = DoCooling_h(rhoGrid[i], uGrid[j], dt_t, XH)
		delta_u = uGrid[j] - ux
		
		res[k, 0] = uGrid[j] # u_ad ====> u before cooling
		res[k, 1] = rhoGrid[i]
		res[k, 2] = dt_t
		res[k, 3]= delta_u
		
		k+=1

print('Elapsed time = ', time.time() - TA)

with open('coolingGrid.pkl', 'wb') as f:
	pickle.dump(res, f)








