
# modified to be used with any number of CPUs.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx2_2t import *


np.random.seed(42)

#---- Constants -----------
eta = 0.1
gama = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#--------------------------
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


with open('Evrard_11536.pkl', 'rb') as f:   # !!!!!! Change epsilon
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
print('T_h1 = ', time.time() - Th1)

Th2 = time.time()
#--------- h (main) ---------
h = smoothing_fast_new_cpp(r, h)
#----------------------------
print('T_main_h = ', time.time() - Th2)

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

TA = time.time()

while t < tEnd:

	TB = time.time()

	#--------- v ----------
	v += acc * dt/2.0
	#----------------------
	
	#--------- r ----------
	r += v * dt
	#----------------------

	T_hh = time.time()
	#--------- h ----------
	h = h_smooth_fast(r, h)
	#----------------------
	print('T_main_h = ', time.time() - T_hh)
	
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
	
	#if not (ii%50):
	#	print('h/c = ', np.sort(h/c))

	#print('Loop time (1) = ', time.time() - TB)
	print()

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	#print('Loop time (2) = ', time.time() - TB)

print('elapsed time = ', time.time() - TA)




