
# New h algorithm is employed !
# The difference with augsphx6.4.py is that here we also use epsilonij instead of epsilon
# The difference with augsphx6.3.py is that here we use hij instead of h

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
from libsx import *
from numba import jit, njit
import concurrent.futures


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


with open('Evrard_1472.pkl', 'rb') as f:   # !!!!!! Change epsilon
    res = pickle.load(f)
resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))

print('The file is read .....')
print()

rSPH = np.hstack((resx, resy, resz))
rDM = rSPH.copy()
N = len(rSPH)

epsilonSPH = np.zeros(N) + 0.10
#epsilonDM = np.zeros((1, N)) + 0.20
epsilon = epsilonSPH #np.hstack((epsilonSPH, epsilonDM))


MSPH = 1.0 # total gas mass
#MDM = 0.9 # total DM mass

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

vSPH = np.hstack((vx.T, vy.T, vz.T))
vDM = vSPH.copy()

uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

#h = smooth_h(rSPH)                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
h = do_smoothingX((rSPH, rSPH))  # This plays the role of the initial h so that the code can start !
h = h_smooth_fast(rSPH, h)
mSPH = np.zeros(N) + MSPH/N
#mDM = np.zeros((1, N)) + MDM/N

r = rSPH #np.vstack((rSPH, rDM))
v = 0.0 * vSPH #np.vstack((vSPH, vDM))
m = mSPH #np.hstack((mSPH, mDM))

rho = getDensity(r, m, h)

TG = time.time()
acc_g = getAcc_g_smthx(r, m, G, epsilon)
print('TG = ', time.time() - TG)

P = getPressure(rho, u, gama)
c = np.sqrt(gama * (gama - 1.0) * u)
#PIij = PI_ij(r, v, rho, c, m, h, eta, alpha, beta)
acc_sph = getAcc_sph(rSPH, vSPH, rho, P, c, h, m, gama, eta, alpha, beta)
acc = acc_g.copy()
acc = acc + acc_sph

t_all = np.arange(Nt+1)*dt

t = 0.0

u_previous = u.copy()
uold = u.copy()
ut = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
ut_previous = ut.copy()

TA = time.time()

i = 0

while t < tEnd:

	TB = time.time()
	
	v += acc * dt/2.0

	r += v * dt

	T1 = time.time()
	h = h_smooth_fast(r, h)
	print('T1 = ', time.time() - T1)

	
	T2 = time.time()
	rho = getDensity(r, m, h)
	print('T2 = ', time.time() - T2)

	#TG1 = time.time()
	#acc_g = getAcc_g_smth(r, m, G, epsilon)
	#print('TG1 = ', time.time() - TG1)

	TG = time.time()
	acc_g = getAcc_g_smthx(r, m, G, epsilon)
	print('TG = ', time.time() - TG)
	
	
	TP = time.time()
	P = getPressure(rho, u, gama)
	c = np.sqrt(gama * (gama - 1.0) * u)

	print('TP = ', time.time() - TP)
	T4 = time.time()
	ut = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	print('T4 = ', time.time() - T4)
	#uold += dt * ut
	#u = u_previous + 0.5 * dt * (ut + ut_previous)
	u += dt * ut

	#u_previous = u.copy()
	#ut_previous = ut.copy()

	#T5 = time.time()
	#acc_sph = getAcc_sph(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	#print('T5 = ', time.time() - T5)

	acc = acc_g.copy()
	#acc = acc + acc_sph

	v += acc * dt/2.0
	
	print('Current time = ', t)
	print('sorted h/c = ', np.sort(h/c))

	t += dt
	
	i += 1

	dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho, 'P': P, 'c': c}
	with open('./Outputs/' + str(i).zfill(5) + '.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	print('Loop time = ', time.time() - TB)
	print()


print('elapsed time = ', time.time() - TA)




