
# modified to be used with any number of CPUs.
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
from mpi4py import MPI


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

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
acc_g = getAcc_g_smth(r, m, G, epsilon)
print('TG = ', time.time() - TG)

P = getPressure(rho, u, gama)
c = np.sqrt(gama * (gama - 1.0) * u)
acc_sph = getAcc_sph(rSPH, vSPH, rho, P, c, h, m, gama, eta, alpha, beta)
acc = acc_g.copy()
acc = acc + acc_sph

t_all = np.arange(Nt+1)*dt

t = 0.0

u_previous = u.copy()
#uold = u.copy()
ut = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
ut_previous = ut.copy()

TA = time.time()

ii = 0

N = r.shape[0]

count = N // nCPUs ##### Note: for now nCPUs MUST be 4 because 5616 is divisible by 4 !!!!!!
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count


while t < tEnd:


	#--------- v ----------
	if rank == 0:
		v += acc * dt/2.0
	
	v = comm.bcast(v, root = 0)
	#----------------------

	#--------- r ----------
	if rank == 0:
		r += v * dt
	
	r = comm.bcast(r, root = 0)
	#----------------------

	#--------- h ----------
	local_h = h_smooth_fast_mpi(nbeg, nend, r, h)
	
	if rank == 0:
		h = local_h
		for i in range(1, nCPUs):
			htmp = comm.recv(source = i)
			h = np.concatenate((h, htmp))
	else:
		comm.send(local_h, dest = 0)

	h = comm.bcast(h, root = 0)
	#----------------------	
	
	#-------- rho ---------
	local_rho = getDensity_mpi(nbeg, nend, r, m, h)
	
	if rank == 0:
		rho = local_rho
		for i in range(1, nCPUs):
			rhotmp = comm.recv(source = i)
			rho = np.concatenate((rho, rhotmp))
	else:
		comm.send(local_rho, dest = 0)
	
	rho = comm.bcast(rho, root = 0)	
	#----------------------
	
	#------- acc_g --------
	local_acc_g = getAcc_g_smth_mpi(nbeg, nend, r, m, G, epsilon)
	
	if rank == 0:
		acc_g = local_acc_g
		for i in range(1, nCPUs):
			acc_gtmp = comm.recv(source = i)
			acc_g = np.concatenate((acc_g, acc_gtmp))
	else:
		comm.send(local_acc_g, dest = 0)
	
	acc_g = comm.bcast(acc_g, root = 0)
	#----------------------	

	#--------- P ----------
	if rank == 0:
		P = getPressure(rho, u, gama)
	
	P = comm.bcast(P, root = 0)
	#----------------------

	#--------- c ----------
	if rank == 0:
		c = np.sqrt(gama * (gama - 1.0) * u)
	
	c = comm.bcast(c, root = 0)
	#----------------------

	#ut1 = get_dU(r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	
	#--------- ut ---------
	local_ut = get_dU_mpi(nbeg, nend, r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	
	if rank == 0:
		ut = local_ut
		for i in range(1, nCPUs):
			ut_tmp = comm.recv(source = i)
			ut = np.concatenate((ut, ut_tmp))
	else:
		comm.send(local_ut, dest = 0)
	
	ut = comm.bcast(ut, root = 0)
	#----------------------

	#--------- u ----------
	#if rank == 0:
	#	u += dt * ut
	#u = comm.bcast(u, root=0)
	#----------------------
	
	#--------- u ----------
	if rank == 0:
		u = u_previous + 0.5 * dt * (ut + ut_previous)

		u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
		ut_previous = ut.copy()
	
	u = comm.bcast(u, root=0)
	u_previous = comm.bcast(u_previous, root=0)
	ut_previous = comm.bcast(ut_previous, root=0)
	#----------------------
	

	#------ acc_sph -------
	local_acc_sph = getAcc_sph_mpi(nbeg, nend, r, v, rho, P, c, h, m, gama, eta, alpha, beta)
	
	if rank == 0:
		acc_sph = local_acc_sph
		for i in range(1, nCPUs):
			acc_sph_tmp = comm.recv(source = i)
			acc_sph = np.concatenate((acc_sph, acc_sph_tmp))
	else:
		comm.send(local_acc_sph, dest = 0)
	
	acc_sph = comm.bcast(acc_sph, root = 0)
	#----------------------

	#-------- acc ---------
	if rank == 0:
		acc = acc_g + acc_sph
	
	acc = comm.bcast(acc, root = 0)
	#----------------------
	
	#--------- v ----------
	if rank == 0:
		v += acc * dt/2.0
	
	v = comm.bcast(v, root = 0)
	#----------------------
	
	t += dt
	
	if rank == 0:
		if not (ii%50):
			print('h/c = ', np.sort(h/c))

	if rank == 0:
		ii += 1
		dictx = {'pos': r, 'v': v, 'm': m, 'uDirectcool': u, 'dt': dt, 'current_t': t, 'rho': rho}
		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	


print('elapsed time = ', time.time() - TA)




