
# Shear viscosity correction (i.e. Balsara switsch) is incorporated.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *
from shear import *
from numba import njit


np.random.seed(42)

M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 0.84 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6

print('unitTime_in_Myr = ', unitTime_in_Myr)
print('unitTime_in_yr = ', unitTime_in_yr)


#---- Constants -----------
eta = 0.1
gamma = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.002
tEnd = 8.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


T_cld = 54.   #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!
T_ps  = T_cld#184. #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!! Calculated from jump condition.
T_0 = 10. #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!

minimum_h = 0.01 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

with open('Data_Uniform_Initial_rho.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
v = data['v'] / unitVelocity
h = data['h']
m = data['m']

print('The file is read .....')
print()


plt.scatter(r[:, 0], r[:, 1], s = 1, color = 'k')
plt.show()


#r = np.hstack((resx, resy, resz))
N = r.shape[0]

epsilon = h.copy() #np.zeros(N) + 0.10

#MSPH = 1.0 # total gas mass

#v = np.zeros_like(r)

#uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
#u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

#Th1 = time.time()
#-------- h (initial) -------
#h = do_smoothingX((r, r))  # This plays the role of the initial h so that the code can start !
#----------------------------
#print('Th1 = ', time.time() - Th1)


print('h = ', np.sort(h))
print(h.shape)


Th2 = time.time()
#--------- h (main) ---------
h = h_smooth_fast_h_minimum_set(r, h, minimum_h)
#----------------------------
print('Th2 = ', time.time() - Th2)

print('h = ', np.sort(h))



#m = np.zeros(N) + MSPH/N

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

#----- P_polytrop
@njit
def P_polytrop(rho, T_cld, T_ps, T_0):

	N = len(rho)
	P_res = np.zeros(N)
	mH = 1.6726e-24 # gram
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH2 = 2.7 * mH
	const = kB/mH2
	
	for i in range(N):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		if rhot <= 1.e-21:
			P_res[i] = rhot * const * T_cld
		
		elif (rhot > 1.e-21) and (rhot <= 5.e-21):
			P_res[i] = rhot * const * gamma * T_cld * (rhot/5e-21)**(gamma-1.)
		
		elif (rhot > 5.e-21) and (rhot <= 1.e-18):
			P_res[i] = rhot * const * T_ps
		
		elif rhot > 1.e-18:
			P_res[i] = rhot * const * T_0 * (1. + gamma * (rhot/1e-14))**(gamma-1.)
	
	P_res = P_res / Unit_P_in_cgs

	return P_res



#----- sound_speed
@njit
def sound_speed(rho, T_cld, T_ps, T_0):

	N = len(rho)
	c = np.zeros(N)
	mH = 1.6726e-24 # gram
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH2 = 2.7 * mH
	const = kB/mH2
	
	for i in range(N):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		c[i] = (const * T_cld)**0.5
		
		if (rhot > 5.e-21) and (rhot <= 1.e-18):
			c[i] = (const * T_ps)**0.5
		
		elif rhot > 1.e-18:
			c[i] = (const * T_0)**0.5
		
	return c / unitVelocity
	

#--------- P ----------
#P = getPressure(rho, u, gamma)
P = P_polytrop(rho, T_cld, T_ps)
#----------------------

#--------- c ----------
c = sound_speed(rho, T_cld, T_ps)
#c = np.sqrt(gamma * (gamma - 1.0) * u)
#----------------------

print('c = ', np.sort(c))

#--- divV & curlV -----
divV, curlV = div_curlVel(r, v, rho, m, h)
#----------------------

#------ acc_sph -------
acc_sph = getAcc_sph_shear(r, v, rho, P, c, h, m, divV, curlV, alpha)
#----------------------

#-------- acc ---------
acc = acc_g + acc_sph
#----------------------

#--------- ut ---------
#ut = get_dU_shear(r, v, rho, P, c, h, m, divV, curlV, alpha)
#----------------------

#--------- u ----------
#u += ut * dt
#u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
#ut_previous = ut.copy()
#----------------------

t = 0.0
ii = 0

TA = time.time()

print('I got here !!!')

while t < tEnd:

	TB = time.time()

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------

	#--------- r ----------
	r += v * dt
	#----------------------
	
	#dictx = {'r': r, 'v': v, 'h': h}
	#with open('debug_Data2.pkl', 'wb') as f:
	#	pickle.dump(dictx, f)
	
	#plt.scatter(r[:, 0], r[:, 1], s = 1, color = 'k')
	#plt.show()
	
	#--------- h ----------
	h = h_smooth_fast_h_minimum_set(r, h, minimum_h)
	#----------------------
		
	#-------- rho ---------
	rho = getDensity(r, m, h)
	#----------------------
	
	epsilon = h.copy()
	
	#------- acc_g --------
	acc_g = getAcc_g_smth(r, m, G, epsilon)
	#----------------------

	#--------- P ----------
	#P = getPressure(rho, u, gamma)
	P = P_polytrop(rho, T_cld, T_ps)
	#----------------------

	#--------- c ----------
	c = sound_speed(rho, T_cld, T_ps)
	#c = np.sqrt(gamma * (gamma - 1.0) * u)
	#----------------------

	#--- divV & curlV -----
	divV, curlV = div_curlVel(r, v, rho, m, h)
	#----------------------
	
	#------ acc_sph -------
	acc_sph = getAcc_sph_shear(r, v, rho, P, c, h, m, divV, curlV, alpha)
	#----------------------

	#-------- acc ---------
	acc = acc_g + acc_sph
	#----------------------
	
	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	#--------- ut ---------
	#ut = get_dU_shear(r, v, rho, P, c, h, m, divV, curlV, alpha)
	#----------------------

	#--------- u ----------
	#u = u_previous + 0.5 * dt * (ut + ut_previous)
	#u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	#ut_previous = ut.copy()
	#----------------------
	
	t += dt
	
	#if True:
	if not (ii%50):
		print('h/c = ', np.sort(h/c))

	print('Loop time (1) = ', time.time() - TB)

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': 0, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time (2) = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




