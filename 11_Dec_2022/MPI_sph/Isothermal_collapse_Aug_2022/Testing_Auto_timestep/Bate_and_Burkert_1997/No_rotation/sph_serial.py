
# modified to be used with any number of CPUs.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *


#----- P_polytrop_mpi
@njit
def getPressure(rho, rho_crit, c_0):

	M = len(rho)
	P_res = np.zeros(M)
	
	for i in range(M):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		P_res[i] = rhot * c_0*c_0 * (1. + (rhot/rho_crit)**(2./3.))
	
	P_res = P_res / Unit_P_in_cgs

	return P_res



@njit
def sound_speed(rho, rho_crit, c_0):

	M = len(rho)
	c = np.zeros(M)
	
	for i in range(M):
		
		rhot = rho[i]*UnitDensity_in_cgs
		c_s2 = c_0*c_0 * (1. + (rhot/rho_crit)**(2./3.))

		c[i] = (c_s2)**0.5
		
	return c / unitVelocity


np.random.seed(42)

M_sun = 1.98992e+33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 1.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!

UnitRadius_in_cm = 9.2e16  #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6

print('unitTime_in_Myr = ', unitTime_in_Myr)

#---- Constants -----------
eta = 0.1
gamma = 5.0/3.0
alpha = 1.0  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
beta = 2.0 * alpha # 1.0   # !!!!!!!!!!!!!!!!!!!!!!!!!!!
G = 1.0
#---------------------------
t = 0.0
dt = 0.0002
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)

minimum_h = 0.0001 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass



with open('hfv_IC_RND_4k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
v = data['v'] #/ unitVelocity
h = data['h']
m = data['m']

unitVelocity = data['unitVelocity']
print('unitVelocity (See hfv_IC_generator) = ', unitVelocity)

c_0 = 1.9e4  #(kB * T_0 / mH2)**0.5   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
rho_crit = 1e-13 # g/cm3

epsilon = h.copy()

v = np.zeros_like(r)

#uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
#u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

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

print(f'c_0 = {c_0}')

print(h.shape)

print(np.sort(h))

print(r.shape)

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
P = getPressure(rho, rho_crit, c_0)
#----------------------

print('P = ', P)
print('*****************')
print()
print('rho = ', rho)
print('*****************')
print()
print('rho_crit = ', rho_crit)
print('*****************')
print()
print('c_0 = ', c_0)
s()

#--------- c ----------
c = sound_speed(rho, rho_crit, c_0)
#----------------------

#------ acc_sph -------
acc_sph = getAcc_sph(r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
#----------------------

print('r = ', r)
print('*****************')
print()
print('v = ', v)
print('*****************')
print()
print('rho = ', rho)
print('*****************')
print()
print('P = ', P)
print('*****************')
print()
print('c = ', c)
print('*****************')
print()
print('h = ', h)
print('*****************')
print()
print('m = ', m)
print('*****************')
print()
print('gamma = ', gamma)
print('*****************')
print()
print('eta = ', eta)
print('*****************')
print()
print('alpha = ', alpha)
print('*****************')
print()
print('beta = ', beta)
s()

#-------- acc ---------
acc = acc_g + acc_sph
#----------------------

#--------- ut ---------
#ut = get_dU(r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
#----------------------

#--------- u ----------
#u += ut * dt
#u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
#ut_previous = ut.copy()
#----------------------

t = 0.0
ii = 0

TA = time.time()

while t < tEnd:

	TB = time.time()
	
	#print(acc)
	#s()

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	plt.scatter(r[:, 0], r[:, 1], s = 0.1, color = 'black')
	plt.show()
	
	#--------- r ----------
	r += v * dt
	#----------------------
	
	plt.scatter(r[:, 0], r[:, 1], s = 0.1, color = 'lime')
	plt.show()
	

	print('h before = ', h)

	#--------- h ----------
	h = h_smooth_fast(r, h)
	#----------------------
	
	print('h after = ', h)
	s()
	
	#-------- rho ---------
	rho = getDensity(r, m, h)
	#----------------------
	
	#------- acc_g --------
	acc_g = getAcc_g_smth(r, m, G, epsilon)
	#----------------------

	#--------- P ----------
	P = getPressure(rho, rho_crit, c_0)
	#----------------------

	#--------- c ----------
	c = sound_speed(rho, rho_crit, c_0)
	#----------------------

	#--------- ut ---------
	#ut = get_dU(r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
	#----------------------

	#--------- u ----------
	#u = u_previous + 0.5 * dt * (ut + ut_previous)
	#u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	#ut_previous = ut.copy()
	#----------------------
	
	#------ acc_sph -------
	acc_sph = getAcc_sph(r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
	#----------------------

	#-------- acc ---------
	acc = acc_g + acc_sph
	#----------------------
	
	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	t += dt
	
	if not (ii%10):
		print('h/c = ', np.sort(h/c))
		print('h = ', np.sort(h))
		print('c = ', np.sort(c))

	print('Loop time (1) = ', time.time() - TB)

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time (2) = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




