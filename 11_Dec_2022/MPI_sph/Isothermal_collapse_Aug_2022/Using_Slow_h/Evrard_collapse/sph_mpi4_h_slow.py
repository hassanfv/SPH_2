
# The difference with _v3 is that here the particles can have different mass and the gravitational acceleration calculation is accordingly modified.
# The difference with _v2 is that here we incorporate shear viscosity by using the Balsara switch.
# The difference with previous version is that here we separated u and u_previous, ut_previous updates separately. See below.
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx import *
from mpi4py import MPI
from shear_test import *


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


#*****************************************************
#************* Reading Input File ********************
#*****************************************************
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



#---- Constants -----------
G = 1.0
eta = 0.1
gamma = 5.0/3.0
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
alpha = 1.0  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
beta = 2.0 * alpha  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
#---------------------------

#----------------------------------------------------
t = 0.0
dt = 0.001
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


print('r.shape = ', r.shape)

N = r.shape[0]

#------- used in MPI --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#----------------------------

#MSPH = 1.0 # total gas mass

MSPH = 1.0
m = np.zeros(N) + MSPH/N

v = np.zeros_like(r)

uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

#if rank == 0:
#	Th1 = time.time()
#-------- h (initial) -------
#local_h = smoothing_length_mpi(nbeg, nend, r)  # This plays the role of the initial h so that the code can start !
#h = 0 # This is just a placeholder. Its absence would crash this line :h = comm.bcast(h, root = 0) for CPUs other than rank = 0.

#if rank == 0:
#	h = local_h
#	for i in range(1, nCPUs):
#		h_tmp = comm.recv(source = i)
#		h = np.concatenate((h, h_tmp))
#else:
#	comm.send(local_h, dest = 0)
#
#h = comm.bcast(h, root = 0)
#----------------------------
#if rank == 0:
#	print('Th1 = ', time.time() - Th1)


Th2 = time.time()
#--------- h (main) ---------
#local_h = h_smooth_fast_mpi(nbeg, nend, r, h)
#local_h = h_smooth_fast_mpi_min_h_set(nbeg, nend, r, h, minimum_h)
local_h = smoothing_length_mpi(nbeg, nend, r)
h = 0.0

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		htmp = comm.recv(source = i)
		h = np.concatenate((h, htmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)
comm.Barrier()
if rank == 0:
	print('Th2 = ', time.time() - Th2)
#----------------------------

epsilon = h.copy()

#m = np.zeros(N) + MSPH/N

#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = getDensity_mpi(nbeg, nend, r, m, h)
rho = 0.0 # This is just a placeholder. Its absence would crash this line :rho = comm.bcast(rho, root = 0) for CPUs other than rank = 0.

if rank == 0:
	rho = local_rho
	for i in range(1, nCPUs):
		rhotmp = comm.recv(source = i)
		rho = np.concatenate((rho, rhotmp))
else:
	comm.send(local_rho, dest = 0)

rho = comm.bcast(rho, root = 0)
	
if rank == 0:
	print('Trho = ', time.time() - Trho)
#----------------------

#----- acc_g_mimj -----
if rank == 0:
	TG = time.time()

local_acc_g = getAcc_g_smth_mimj_mpi(nbeg, nend, r, m, G, epsilon)
acc_g = 0.0

if rank == 0:
	acc_g = local_acc_g
	for i in range(1, nCPUs):
		acc_gtmp = comm.recv(source = i)
		acc_g += acc_gtmp
else:
	comm.send(local_acc_g, dest = 0)

acc_g = comm.bcast(acc_g, root = 0)

if rank == 0:
	print('TG = ', time.time() - TG)
#----------------------

abs_acc = (acc_g[:, 0]*acc_g[:, 0] + acc_g[:, 1]*acc_g[:, 1] + acc_g[:, 2]*acc_g[:, 2])**0.5
dt_f = np.min((h/abs_acc)**0.5)
#ErrTolIntAccuracy = 0.25
#dt_f2 = np.min((2.*ErrTolIntAccuracy*h/abs_acc)**0.5)


#--------- P ----------
P = 0.0
if rank == 0:
	P = getPressure(rho, u, gamma)

P = comm.bcast(P, root = 0)
#----------------------

#--------- c ----------
c = 0.0
if rank == 0:
	c = np.sqrt(gamma * (gamma - 1.0) * u)

c = comm.bcast(c, root = 0)
#----------------------


#----- div_curlV ------
divV = 0.0
curlV = 0.0
local_div_curlV = div_curlVel_mpi(nbeg, nend, r, v, rho, m, h)

if rank == 0:
	divV, curlV = local_div_curlV
	for i in range(1, nCPUs):
		divV_tmp, curlV_tmp = comm.recv(source = i)
		divV = np.concatenate((divV, divV_tmp))
		curlV = np.concatenate((curlV, curlV_tmp))
else:
	comm.send(local_div_curlV, dest = 0)

divV = comm.bcast(divV, root = 0)
curlV = comm.bcast(curlV, root = 0)
#----------------------

#------ acc_sph mu_visc -------
acc_sph = 0.0
#dt_cv = 0.0
tmp = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

local_acc_sph = tmp
#local_dt_cv = tmp[1]


if rank == 0:
	acc_sph = local_acc_sph
	#dt_cv = local_dt_cv
	for i in range(1, nCPUs):
		tmpt = comm.recv(source = i)
		acc_sph_tmp = tmpt
#		dt_cv_tmp = tmpt[1]
		acc_sph = np.concatenate((acc_sph, acc_sph_tmp))
#		dt_cv = np.concatenate((dt_cv, dt_cv_tmp))
else:
	comm.send(local_acc_sph, dest = 0)

acc_sph = comm.bcast(acc_sph, root = 0)
#dt_cv = comm.bcast(dt_cv, root = 0)
#dt_cv = np.min(dt_cv)
#----------------------


#-------- acc ---------
acc = 0.0
if rank == 0:
	acc = acc_g + acc_sph

acc = comm.bcast(acc, root = 0)
#----------------------

abs_acc = (acc[:, 0]*acc[:, 0] + acc[:, 1]*acc[:, 1] + acc[:, 2]*acc[:, 2])**0.5
dt_kin = np.min((h/abs_acc)**0.5)

#--------- ut ---------
ut = 0.0
local_ut = get_dU_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

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
u_previous = 0.0
ut_previous = 0.0
if rank == 0:
	u += ut * dt

	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	ut_previous = ut.copy()

u = comm.bcast(u, root=0)
u_previous = comm.bcast(u_previous, root=0)
ut_previous = comm.bcast(ut_previous, root=0)
#----------------------
comm.Barrier()

t = 0.0
ii = 0

TA = time.time()

while t < tEnd:

	if rank == 0:
		TLoop = time.time()

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
	#local_h = h_smooth_fast_mpi(nbeg, nend, r, h)
	#local_h = h_smooth_fast_mpi_min_h_set(nbeg, nend, r, h, minimum_h)
	local_h = smoothing_length_mpi(nbeg, nend, r)
	
	if rank == 0:
		h = local_h
		for i in range(1, nCPUs):
			htmp = comm.recv(source = i)
			h = np.concatenate((h, htmp))
	else:
		comm.send(local_h, dest = 0)

	h = comm.bcast(h, root = 0)
	#----------------------
	
	#---- Setting Minimum h ---
	#if rank == 0:
	#	nnx = np.where(h < minimum_h)[0]
	#	h[nnx] = minimum_h
	
	#h = comm.bcast(h, root = 0)
	#--------------------------
	
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
	
	#epsilon = h.copy()
	epsilon = np.zeros_like(h) + 0.10
	
	#----- acc_g_mimj -----
	local_acc_g = getAcc_g_smth_mimj_mpi(nbeg, nend, r, m, G, epsilon)
	acc_g = 0.0

	if rank == 0:
		acc_g = local_acc_g
		for i in range(1, nCPUs):
			acc_gtmp = comm.recv(source = i)
			acc_g += acc_gtmp
	else:
		comm.send(local_acc_g, dest = 0)

	acc_g = comm.bcast(acc_g, root = 0)
	#----------------------

	#--------- P ----------
	if rank == 0:
		P = getPressure(rho, u, gamma)

	P = comm.bcast(P, root = 0)
	#----------------------

	#--------- c ----------
	if rank == 0:
		c = np.sqrt(gamma * (gamma - 1.0) * u)
	
	c = comm.bcast(c, root = 0)
	#----------------------

	#----- div_curlV ------
	local_div_curlV = div_curlVel_mpi(nbeg, nend, r, v, rho, m, h)

	if rank == 0:
		divV, curlV = local_div_curlV
		for i in range(1, nCPUs):
			divV_tmp, curlV_tmp = comm.recv(source = i)
			divV = np.concatenate((divV, divV_tmp))
			curlV = np.concatenate((curlV, curlV_tmp))
	else:
		comm.send(local_div_curlV, dest = 0)

	divV = comm.bcast(divV, root = 0)
	curlV = comm.bcast(curlV, root = 0)
	#----------------------

	#------ acc_sph mu_visc -------
	acc_sph = 0.0
	#dt_cv = 0.0
	tmp = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

	local_acc_sph = tmp
	#local_dt_cv = tmp[1]


	if rank == 0:
		acc_sph = local_acc_sph
	#	dt_cv = local_dt_cv
		for i in range(1, nCPUs):
			tmpt = comm.recv(source = i)
			acc_sph_tmp = tmpt
	#		dt_cv_tmp = tmpt[1]
			acc_sph = np.concatenate((acc_sph, acc_sph_tmp))
	#		dt_cv = np.concatenate((dt_cv, dt_cv_tmp))
	else:
		comm.send(local_acc_sph, dest = 0)

	acc_sph = comm.bcast(acc_sph, root = 0)
	#dt_cv = comm.bcast(dt_cv, root = 0)
	#dt_cv = np.min(dt_cv)
	#----------------------

	#-------- acc ---------
	if rank == 0:
		acc = acc_g + acc_sph
	
	acc = comm.bcast(acc, root = 0)
	#----------------------
	
	#--- TimeStep Calculation ---
	abs_acc = (acc_g[:, 0]*acc_g[:, 0] + acc_g[:, 1]*acc_g[:, 1] + acc_g[:, 2]*acc_g[:, 2])**0.5
	dt_f = np.min((h/abs_acc)**0.5)
	
	abs_acc = (acc[:, 0]*acc[:, 0] + acc[:, 1]*acc[:, 1] + acc[:, 2]*acc[:, 2])**0.5
	dt_kin = np.min((h/abs_acc)**0.5)

	#---
	v_sig = [np.max(c[i]+c) for i in range(len(c))]
	v_signal = np.max(v_sig)
	C_CFL = 0.25
	dt_cfl = np.min(C_CFL * h / v_signal)
	#---
	
	dh_dt = 1./3. * h * divV # See the line below eq.31 in Gadget 2 paper.
	dt_dens = np.min(C_CFL * h / np.abs(dh_dt))

	dt = 0.25 * np.min([dt_f, dt_kin, dt_cfl, dt_dens])
	
	if dt < 5e-5:
		dt = 5e-5 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	dt = 0.001
	
	#if dt > 0.001:
	#	dt = 0.001
	#dt = 1e-4 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	#--- End of TimeStep Calculation --
	
	#--------- v ----------
	if rank == 0:
		v += acc * dt/2.0
	
	v = comm.bcast(v, root = 0)
	#----------------------
	
	#--------- ut ---------
	local_ut = get_dU_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)
	
	if rank == 0:
		ut = local_ut
		for i in range(1, nCPUs):
			ut_tmp = comm.recv(source = i)
			ut = np.concatenate((ut, ut_tmp))
	else:
		comm.send(local_ut, dest = 0)
	#
	ut = comm.bcast(ut, root = 0)
	#----------------------

	#--------- u ----------
	if rank == 0:
		u = u_previous + 0.5 * dt * (ut + ut_previous)

	u = comm.bcast(u, root=0)
	#----------------------
	
	# Heating and Cooling implementation comes here !!!!
	
	#----- u previous -----
	if rank == 0:
		u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
		ut_previous = ut.copy()
	
	u_previous = comm.bcast(u_previous, root=0)
	ut_previous = comm.bcast(ut_previous, root=0)
	#----------------------
	
	t += dt
	
	if rank == 0:
		if not (ii%50):
			print('h/c = ', np.sort(h/c))
			print('h = ', np.sort(h))

	if rank == 0:
		ii += 1
		dictx = {'pos': r, 'v': v, 'u': u, 'm': m, 'dt': dt, 'current_t': t, 'rho': rho, 'h': h}

		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	
	if rank == 0:
		print('dt_f, dt_kin, dt_cfl, dt_dens = ', [dt_f, dt_kin, dt_cfl, dt_dens])
		print(f'Adopted dt = {dt}')
		print('Loop time = ', time.time() - TLoop)

print('elapsed time = ', time.time() - TA)




