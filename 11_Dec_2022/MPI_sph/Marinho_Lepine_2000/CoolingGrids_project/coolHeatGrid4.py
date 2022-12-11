
import numpy as np
import matplotlib.pyplot as plt
import time
from coolHeat2libs import *
from scipy.interpolate import griddata
import pickle


unit_U = 9537129466.43166
UnitTime_in_s = 3.16e13 # == 1Myr
unit_rho = 1.5e-20 # g/cm3


#***** LambdaGrid
def LambdaGrid(rho, u_vect):

	rho_cgs = rho * unit_rho

	res = []

	for i in range(len(u_vect)):

		#---- calculating some rough T_init ----
		y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
		mu_approx = X/2. + X*y/2. + Y/4.
		T_init = (gamma - 1.0) * mH / kB * mu_approx * u_vect[i] * unit_U
		#---------------------------------------

		res.append([u_vect[i], -coolingRateFromU_M(u_vect[i] * unit_U, T_init, rho_cgs)])


	return np.array(res)



#***** LambdaGrid_Single
def LambdaGrid_Single(rho, u):

	rho_cgs = rho * unit_rho
	u_cgs = u * unit_U

	#---- calculating some rough T_init ----
	y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
	mu_approx = X/2. + X*y/2. + Y/4.
	T_init = (gamma - 1.0) * mH / kB * mu_approx * u_cgs
	#---------------------------------------

	return -coolingRateFromU_M(u_cgs, T_init, rho_cgs)



u_vect = np.logspace(np.log10(0.01), np.log10(13400.), 700) # in code unit
rho_vect = np.logspace(np.log10(1e-7), np.log10(112.), 700) # in code unit


N_u = len(u_vect)
N_rho = len(rho_vect)

LambdaZ = np.zeros(N_u*N_rho)
rhoZ = np.zeros((N_u*N_rho))
uZ = np.zeros((N_u*N_rho))

i = 0

T1 = time.time()

for rho in rho_vect:

	for u in u_vect:
		
		LambdaZ[i] = LambdaGrid_Single(rho, u) # u and rho will be converted to physical units inside the "LambdaGrid_Single" function.
		uZ[i] = u # in code unit
		rhoZ[i] = rho # in code unit
		
		i += 1

print('T1 = ', time.time() - T1)

#-------------
dictx = {'u_vect': u_vect, 'rho_vect': rho_vect, 'uZ': uZ, 'rhoZ': rhoZ, 'LambdaZ': LambdaZ}
with open('u_rho_vects.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#-------------


ut = 17.
rhot = 10.

n11 = np.where((uZ <= ut) & (rhoZ <= rhot))[0]

print(n11)
print()
print(uZ[n11])
print(rhoZ[n11])
print()
print(LambdaZ[n11[-1]])







