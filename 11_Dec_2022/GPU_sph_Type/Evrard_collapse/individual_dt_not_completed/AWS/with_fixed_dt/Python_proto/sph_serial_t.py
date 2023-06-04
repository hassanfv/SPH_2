
# modified to be used with any number of CPUs.
# New h algorithm is employed !


import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *
import readchar

np.random.seed(42)


def create_block_dt_i(dt):

	dt_0 = 0.001
	dt_1 = 0.004
	dt_2 = 0.008
	dt_3 = 0.012
	dt_4 = 0.016
	dt_5 = 0.020

	for i in range(len(dt)):

		if dt[i] <= dt_0:
			dt[i] = dt_0

		if (dt[i] >= dt_0) & (dt[i] < dt_1):
			dt[i] = dt_0

		if (dt[i] >= dt_1) & (dt[i] < dt_2):
			dt[i] = dt_1

		if (dt[i] >= dt_2) & (dt[i] < dt_3):
			dt[i] = dt_2

		if (dt[i] >= dt_3) & (dt[i] < dt_4):
			dt[i] = dt_3

		if (dt[i] >= dt_4) & (dt[i] < dt_5):
			dt[i] = dt_4

		if dt[i] >= dt_5:
			dt[i] = dt_5
		
	return dt
        


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

t += dt #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('Evrard_1472.pkl', 'rb') as f:   # !!!!!! Change epsilon
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


v += acc * dt
r += v * dt


acc2 = (acc[:, 0] * acc[:, 0] + acc[:, 1] * acc[:, 1] + acc[:, 2] * acc[:, 2])
dt_i = (h*h/acc2)**0.25 * 0.004 # 0.02 is my arbitrary multiplication to make dt_i smaller!
print(np.sort(dt_i))
dt_i = create_block_dt_i(dt_i)
print(np.sort(dt_i))

t_last = np.zeros_like(h)
t_next = dt_i.copy()
activeId = np.zeros_like(h)

# Now we check which particles have t_next == t
activeId[(t_next >= t-0.25*dt) & (t_next <= t+0.25*dt)] = 1

print('t = ', t)

print(np.sort(activeId), np.sum(activeId))





#t = 0.0
ii = 0

TA = time.time()

while t < tEnd:

	TB = time.time()
	
	t += dt
	print('t = ', t)

	#--------- v ----------
	#v += acc * dt/2.0
	#----------------------
	
	#--------- r ----------
	#r += v * dt
	#----------------------

	#--------- h ----------
	h = h_smooth_fast(r, h)
	#----------------------
	
	#-------- rho ---------
	rho = getDensity(r, m, h)
	#----------------------
	
	#------- acc_g --------
	acc_g, t_last = getAcc_g_smthxB(r, m, G, epsilon, activeId, t_last, t)
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
	
	
	acc2 = (acc[:, 0] * acc[:, 0] + acc[:, 1] * acc[:, 1] + acc[:, 2] * acc[:, 2])
	dt_i = (h*h/acc2)**0.25 * 0.002 # 0.02 is my arbitrary multiplication to make dt_i smaller!
	print('dt_i (before Block) = ', np.sort(dt_i))
	dt_i = create_block_dt_i(dt_i)
	
	t_next = t_last + dt_i
	
	print('np.sort(t_last) = ', np.sort(t_last))
	print('dt_i = ', np.sort(dt_i))
	
	activeId *= 0
	activeId[(t_next >= t-0.25*dt) & (t_next <= t+0.25*dt)] = 1
	print('N Active = ', np.sum(activeId))
	
	v += acc * dt
	r += v * dt
	
	#kb = readchar.readkey()
	#if kb == 'q':
	#	break

	ii += 1
	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
	with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time = ', time.time() - TB)
	print()

print('elapsed time = ', time.time() - TA)




