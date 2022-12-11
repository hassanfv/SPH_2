
import numpy as np
from coolHeat2libs import *

#--- Scaling From Marinho et al. (2000)-----
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
#-------------------------------------------

u_vect = np.logspace(np.log10(0.05), np.log10(1340.),  50) # in code unit
rho_vect = np.logspace(np.log10(1e-7), np.log10(112.), 50) # in code unit


dt = 0.005

N = len(u_vect)

res = []

TA = time.time()

k = 0

res = np.zeros((N*N, 4))

for i in range(N): # u loop

	#TA = time.time()
	for j in range(N): # rho loop

		ut_cgs = u_vect[i] * Unit_u_in_cgs
		rhot_cgs = rho_vect[j] * UnitDensity_in_cgs
		dt_cgs = dt * UnitTime_in_s

		ucool = DoCooling_M(rhot_cgs, ut_cgs, dt_cgs) / Unit_u_in_cgs
		
		#res.append([rho_vect[j], u_vect[i], ucool, dt])
		res[k, 0] = rho_vect[j]
		res[k, 1] = u_vect[i]
		res[k, 2] = ucool
		res[k, 3] = dt
		k += 1


with open('u_after_cool.pkl', 'wb') as f:
	pickle.dump(res, f)

print('TA = ', time.time() - TA)







