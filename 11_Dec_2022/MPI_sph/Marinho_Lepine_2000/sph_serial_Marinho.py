
# modified to be used with any number of CPUs.
# New h algorithm is employed !
# The difference with augsphx6.4.py is that here we also use epsilonij instead of epsilon
# The difference with augsphx6.3.py is that here we use hij instead of h

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from numba import njit
from libsx import *
from coolHeat2libs import *


#============= COOLING FROM TABLE ====================
with open('./CoolingGrids_project/u_after_cool_MPI_dt_0.005_N_200.pkl', 'rb') as f:
	data = pickle.load(f)

NN = int(np.sqrt(data.shape[0]))
rhoG = data[:, 0].reshape(NN, NN)
uBefore = data[:, 1].reshape(NN, NN)
uAfter = data[:, 2].reshape(NN, NN)
dt = data[:, 3].reshape(NN, NN)

u_uniq = np.unique(uBefore)
rho_uniq = np.unique(rhoG)

@njit
def do_grid_cooling(ut, rhot):

	N = len(ut)
	ures = np.zeros(N)

	for i in range(N):

		#-------- n11 i.e. lower left -----------
		n11 = np.where(u_uniq <= ut[i])[0]
		u11 = np.max(u_uniq[n11])

		n11 = np.where(rho_uniq <= rhot[i])[0]
		rho11 = np.max(rho_uniq[n11])
		#----------------------------------------

		ndx = np.where((uBefore == u11) & (rhoG == rho11))

		nrow = ndx[0][0]
		ncol = ndx[1][0]

		ures[i] = uAfter[nrow, ncol]

	return ures
#=====================================================



np.random.seed(42)

#--- Scaling From Marinho et al. (2000)---
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print('M = ', UnitMass_in_g)
print('M/M_sun = ', UnitMass_in_g/M_sun)
print('U = ', Unit_u_in_cgs)
print('RHO = ', UnitDensity_in_cgs)
print()

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.005
tEnd = 6.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs_'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('Marinho_IC.pkl', 'rb') as f:
    res = pickle.load(f)

r = res['r']
v = res['v']

print('The file is read .....')
print()

N = r.shape[0]

epsilon = np.zeros(N) + 0.10

MSPH = 10.0 # total gas mass


uFloor = 0.211 # 0.211 corresponds to T = 20 K. I used u_as_a_func_of_T function to calculate it !
u = np.zeros(N) + uFloor

Th1 = time.time()
#-------- h (initial) -------
h = do_smoothingX((r, r))  # This plays the role of the initial h so that the code can start !
#----------------------------
print('Th1 = ', time.time() - Th1)


Th2 = time.time()
#--------- h (main) ---------
h = h_smooth_fast(r, h)
#----------------------------
print('Th2 = ', time.time() - Th2)

m = np.zeros(N) + MSPH/N

#-------- rho ---------
Trho = time.time()
rho = getDensity(r, m, h)
print('Trho = ', time.time() - Trho)
#----------------------

#------- acc_g --------
TG = time.time()
acc_g = getAcc_g_smth(r, m, G, epsilon)
print('TG = ', time.time() - TG)
#----------------------	

#--------- P ----------
P = getPressure(rho, u, gama)
#----------------------

#--------- c ----------
c = np.sqrt(gama * (gama - 1.0) * u)
#----------------------

#------ acc_sph -------
acc_sph = getAcc_sph(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
#----------------------

#-------- acc ---------
acc = acc_g + acc_sph
#----------------------

#--------- ut ---------
ut = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
#----------------------

#--------- u ----------
u += ut * dt
u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
ut_previous = ut.copy()
#----------------------

t = 0.0
ii = 0


def apply_cooling_on_u(rho, u, dt):

	dt_cgs = dt * UnitTime_in_s
	u_cgs = u * Unit_u_in_cgs
	rho_cgs = rho * UnitDensity_in_cgs

	u_after_cooling = np.zeros_like(u)

	for i in range(N):
		#TF = time.time()
		#u_after_cooling[i] = DoCooling_M(rho_cgs[i], u_cgs[i], dt_cgs)
		u_after_cooling[i] = DoCooling_M_Lambda_from_Grid(rho_cgs[i], u_cgs[i], dt_cgs)
		#print('TF = ', time.time() - TF)

	return u_after_cooling / Unit_u_in_cgs



TA = time.time()

while t < tEnd:

	TB = time.time()

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------

	#--------- r ----------
	r += v * dt
	#----------------------

	#--------- h ----------
	h = h_smooth_fast(r, h)
	#----------------------
	
	#-------- rho ---------
	rho = getDensity(r, m, h)
	#----------------------
	
	#------- acc_g --------
	acc_g = getAcc_g_smth(r, m, G, epsilon)
	#----------------------

	#--------- P ----------
	P = getPressure(rho, u, gama)
	#----------------------

	#--------- c ----------
	c = np.sqrt(gama * (gama - 1.0) * u)
	#----------------------

	#--------- ut ---------
	ut = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	#----------------------

	#--------- u ----------
	u = u_previous + 0.5 * dt * (ut + ut_previous)
	
	print()
	TCool = time.time()
	print('u before cooling = ', u[1], u[100], u[1232], u[2634], u[3763])
	u = do_grid_cooling(u, rho)
	u[u < 0.2] = 0.2
	print('u before cooling = ', u[1], u[100], u[1232], u[2634], u[3763])
	print('TCool = ', time.time() - TCool)
	print()

	
	#s()
	
	
	
	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	ut_previous = ut.copy()
	#----------------------
	
	#------ acc_sph -------
	acc_sph = getAcc_sph(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	#----------------------

	#-------- acc ---------
	acc = acc_g + acc_sph
	#----------------------
	
	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	t += dt
	
	if not (ii%50):
		print('h/c = ', np.sort(h/c))

	#print('Loop time (1) = ', time.time() - TB)

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




