
import numpy as np
import matplotlib.pyplot as plt
import time
from coolHeat2 import *
from scipy.interpolate import griddata


unit_U = 9537129466.43166
UnitTime_in_s = 3.16e13 # == 1Myr
unit_rho = 1.5e-20 # g/cm3


def LambdaGrid(rho):

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



def LambdaGrid_Single(rho, u):

	rho_cgs = rho * unit_rho
	u_cgs = u * unit_U

	#---- calculating some rough T_init ----
	y = 0.5 # Just approximation to have some estimate of T_init (doesn't need to be exact).
	mu_approx = X/2. + X*y/2. + Y/4.
	T_init = (gamma - 1.0) * mH / kB * mu_approx * u_cgs
	#---------------------------------------

	return -coolingRateFromU_M(u_cgs, T_init, rho_cgs)



u_vect = np.logspace(np.log10(0.1), np.log10(13400.), 50)
rho_vect = np.logspace(1e-7, 112, 50)

N_u = len(u_vect)
N_rho = len(rho_vect)

LambdaZ = np.zeros(N_u*N_rho)
u_rho = np.zeros((N_u*N_rho, 2))

i = 0

T1 = time.time()

for rho in rho_vect:

	for u in u_vect:
		
		LambdaZ[i] = LambdaGrid_Single(rho, u)
		u_rho[i, 0] = u
		u_rho[i, 1] = rho
		
		i += 1

print('T1 = ', time.time() - T1)

print(u_rho.shape)
print(LambdaZ.shape)


u_vect = np.logspace(0.1, 13400., 100)
rho_vect = np.logspace(1e-7, 112., 100)
u_mesh, rho_mesh = np.meshgrid(u_vect, rho_vect, indexing='ij')
print(u_mesh.shape, rho_mesh.shape)

grid_Lambda = griddata(u_rho, LambdaZ, (u_mesh, rho_mesh), method='cubic')

print(grid_Lambda.shape)










