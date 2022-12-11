
# Shear viscosity correction (i.e. Balsara switsch) is incorporated.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx_v2 import *
from shear_itime import *
from core_sph_func import *


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


with open('Evrard_4224.pkl', 'rb') as f:   # !!!!!! Change epsilon
    res = pickle.load(f)
resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))

print('The file is read .....')
print()

r = np.hstack((resx, resy, resz))
N = r.shape[0]

epsilon = np.zeros(N) + 0.10

MSPH = 1.0 # total gas mass

v = np.zeros_like(r)

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

uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

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
ut = get_dU_shear(r, v, rho, P, c, h, m, divV, curlV, alpha)
#----------------------

#--------- u ----------
u += ut * dt
u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
ut_previous = ut.copy()
#----------------------

t = 0.0
ii = 0

itt = 1

TA = time.time()

while t < tEnd:

	TB = time.time()
	
	if itt == 1:
		#--- TimeStep Calculation ---
		dt_cv = 0.1 #!!! CHECK LATER !!!!
		
		abs_acc = (acc_g[:, 0]*acc_g[:, 0] + acc_g[:, 1]*acc_g[:, 1] + acc_g[:, 2]*acc_g[:, 2])**0.5
		dt_f = (h/abs_acc)**0.5
		
		abs_acc = (acc[:, 0]*acc[:, 0] + acc[:, 1]*acc[:, 1] + acc[:, 2]*acc[:, 2])**0.5
		dt_kin = (h/abs_acc)**0.5

		#---
		v_sig = [np.max(c[i]+c) for i in range(len(c))]
		v_signal = v_sig
		C_CFL = 0.25
		dt_cfl = C_CFL * h / v_signal
		#---
		
		#dh_dt = 1./3. * h * divV # See the line below eq.31 in Gadget 2 paper.
		#dt_dens = np.min(C_CFL * h / np.abs(dh_dt))

		dt_f = dt_f.reshape((-1, 1))
		dt_kin = dt_kin.reshape((-1, 1))
		dt_cfl = dt_cfl.reshape((-1, 1))

		dt_arr = np.hstack((dt_f, dt_kin, dt_cfl))
		
		dtz = np.min(dt_arr, axis = 1) * 0.25 # This is an array! MUST BE min NOT max !

		print('sorted dtz = ', dtz)

		dt_max = np.max(dtz)
		#-------------------------------
		
		nn = 5
		nn = np.arange(nn)
		dt_i = np.sort(dt_max / 2**nn)
		
		#-- Grouping the particles based on their dt	
		nG0 = np.where((dtz < dt_i[0]))[0]
		nG2 = np.where((dtz >= dt_i[0]) & (dtz <  dt_i[1]))[0]
		nG4 = np.where((dtz >= dt_i[1]) & (dtz <  dt_i[2]))[0]
		nG8 = np.where((dtz >= dt_i[2]) & (dtz <  dt_i[3]))[0]
		nG16= np.where((dtz >= dt_i[3]) & (dtz <= dt_i[4]))[0]

		print(len(nG0), len(nG2), len(nG4), len(nG8), len(nG16))
		
		dt = np.min(dtz)
		
		if dt > 0.01:
			dt = 0.01


#=============== GROUP 16 ==================
	if np.mod(itt, 16) == 0:
	
		print('Inside Group 16 !!!')
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG16)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG8)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG4)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG2)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG0)
	
#=============== GROUP 8 ==================
	elif np.mod(itt, 8) == 0:
	
		print('Inside Group 8 !!!')

		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG8)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG4)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG2)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG0)

#=============== GROUP 4 ==================
	elif np.mod(itt, 4) == 0:
	
		print('Inside Group 4 !!!')

		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG4)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG2)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG0)


#=============== GROUP 2 ==================
	elif np.mod(itt, 2) == 0:
	
		print('Inside Group 2 !!!')

		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG2)
	
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG0)

#=============== GROUP 0 ==================
	else: # (i.e. if np.mod(itt, 0) != 0)
	
		print('Inside Group 0 !!!')
		
		r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph, u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt = \
			core_sph_func(r, v, acc, h, m, rho, P, c, acc_g, divV, curlV, acc_sph,\
			u, ut, u_previous, ut_previous, G, epsilon, gama, alpha, dt, nG0)
	itt += 1


	t += dt
	
	if itt > 16:
		itt = 1
	
	print('Adopted dt = ', dt)
	
	#if not (ii%50):
	#	print('h/c = ', np.sort(h/c))

	print('Loop time (1) = ', time.time() - TB)

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	#print('Loop time (2) = ', time.time() - TB)
	
	print()

print('elapsed time = ', time.time() - TA)




