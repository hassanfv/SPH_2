
# Shear viscosity correction (i.e. Balsara switsch) is incorporated.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *
from shear import *


np.random.seed(42)

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.001
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


#with open('Evrard_2176.pkl', 'rb') as f:   # !!!!!! Change epsilon
#    res = pickle.load(f)
#resx = res['x'].reshape((len(res['x']),1))
#resy = res['y'].reshape((len(res['x']),1))
#resz = res['z'].reshape((len(res['x']),1))

with open('Boss_IC_5000.pkl', 'rb') as f:
	dict_r_v = pickle.load(f)

r = dict_r_v['r']
v = dict_r_v['v']


print('The file is read .....')
print()

#r = np.hstack((resx, resy, resz))
N = r.shape[0]

epsilon = np.zeros(N) + 0.08

MSPH = 1.0 # total gas mass

#v = np.zeros_like(r)

calculate_velocity = False
if calculate_velocity:

	rr = np.sqrt(resx**2 + resy**2 + resz**2).reshape((1, N))
	omega = 0.5 # angular velocity.
	vel = rr * omega

	sin_T = resy.T / rr
	cos_T = resx.T / rr

	vx = np.abs(vel * sin_T)
	vy = np.abs(vel * cos_T)
	vz = 0.0 * vx


	nregA = (resx.T >= 0.0) & (resy.T >= 0.0)
	vx[nregA] = -vx[nregA]

	nregB = (resx.T < 0.0) & (resy.T >= 0.0)
	vx[nregB] = -vx[nregB]
	vy[nregB] = -vy[nregB]

	nregC = (resx.T < 0.0) & (resy.T < 0.0)
	vy[nregC] = -vy[nregC]

	v = np.hstack((vx.T, vy.T, vz.T))

uFloor = 0.2332 # corresponding to T = 10 K for a gas of pure molecular hydrogen
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


#----- c in isothermal ----
unitVelocity =  51520.44547555854 # see IC_isothermal.py code
R_gas_in_cgs = 8.314e7 # erg/mol/K
T_gas_in_K = 4. # K
c_iso_1 = np.sqrt(R_gas_in_cgs*T_gas_in_K) / unitVelocity

c_iso_2 = np.sqrt((gama - 1.0) * u)

print('c = ', np.sort(c))
print('c_iso_1 = ', c_iso_1)
print('c_iso_2 = ', c_iso_2)

#--------------------------


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
	
	if not (ii%50):
		print('h/c = ', np.sort(h/c))

	print('Loop time (1) = ', time.time() - TB)

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time (2) = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




