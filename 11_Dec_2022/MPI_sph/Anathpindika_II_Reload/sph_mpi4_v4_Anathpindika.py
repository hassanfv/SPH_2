
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
from shear import *




#----- P_polytrop_mpi
@njit
def P_polytrop_mpi(nbeg, nend, rho, T_cld, T_ps, T_0):

	M = nend - nbeg
	P_res = np.zeros(M)
	mH = 1.6726e-24 # gram
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH2 = 2.7 * mH
	const = kB/mH2
	
	for i in range(nbeg, nend):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		if rhot <= 1.e-21:
			P_res[i-nbeg] = rhot * const * T_cld
		
		elif (rhot > 1.e-21) and (rhot <= 5.e-21):
			P_res[i-nbeg] = rhot * const * gamma * T_cld * (rhot/5.e-21)**(gamma-1.)
		
		elif (rhot > 5.e-21) and (rhot <= 1.e-18):
			P_res[i-nbeg] = rhot * const * T_ps
		
		elif rhot > 1.e-18:
			P_res[i-nbeg] = rhot * const * T_0 * (1. + gamma * (rhot/1e-14)**(gamma-1.))
	
	P_res = P_res / Unit_P_in_cgs

	return P_res



#----- sound_speed_mpi
@njit
def sound_speed_mpi(nbeg, nend, rho, T_cld, T_ps, T_0):

	M = nend - nbeg
	c = np.zeros(M)
	mH = 1.6726e-24 # gram
	kB = 1.3807e-16  # cm2 g s-2 K-1
	mH2 = 2.7 * mH
	const = kB/mH2
	
	for i in range(nbeg, nend):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		c[i-nbeg] = (const * T_cld)**0.5
		
		if (rhot > 5.e-21) and (rhot <= 1.e-18):
			c[i-nbeg] = (const * T_ps)**0.5
		
		elif rhot > 1.e-18:
			c[i-nbeg] = (const * T_0)**0.5
		
	return c / unitVelocity



np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
#rB = 0.8 # pc
#ksi = 3.
R_0 = 0.84 #rB/ksi
UnitRadius_in_cm = R_0 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6

print('unitTime_in_Myr = ', unitTime_in_Myr)
print('unitVelocity = ', unitVelocity)

T_cld = 54.   #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!
T_ps  = T_cld#184. #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!! Calculated from jump condition.
T_0 = 10. #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!

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



with open('Data_Turner.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
v = data['v'] / unitVelocity
h = data['h']

print(h.shape)

print(np.sort(h))

print(r.shape)


#if rank == 0:
#	print(np.sort(v.flatten()))

#	import matplotlib.pyplot as plt
#	plt.scatter(r[:, 0], r[:, 1], s = 1, color = 'k')
#	plt.show()


print('The file is read .....')
print()

#r = np.hstack((resx, resy, resz))
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

MSPH = 1.0 # total gas mass

#v = np.zeros_like(r)

#uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
#u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

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
local_h = h_smooth_fast_mpi(nbeg, nend, r, h)

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

m = np.zeros(N) + MSPH/N

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


#--------- p ----------
P = 0.
local_P = P_polytrop_mpi(nbeg, nend, rho, T_cld, T_ps, T_0)

if rank == 0:
	P = local_P
	for i in range(1, nCPUs):
		Ptmp = comm.recv(source = i)
		P = np.concatenate((P, Ptmp))
else:
	comm.send(local_P, dest = 0)

P = comm.bcast(P, root = 0)
#----------------------

#--------- c ----------
c = 0.
local_c = sound_speed_mpi(nbeg, nend, rho, T_cld, T_ps, T_0)

if rank == 0:
	c = local_c
	for i in range(1, nCPUs):
		ctmp = comm.recv(source = i)
		c = np.concatenate((c, ctmp))
else:
	comm.send(local_c, dest = 0)

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

#------ acc_sph -------
acc_sph = 0.0
local_acc_sph = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha, beta, eta)

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
acc = 0.0
if rank == 0:
	acc = acc_g + acc_sph

acc = comm.bcast(acc, root = 0)
#----------------------

#--------- ut ---------
#ut = 0.0
#local_ut = get_dU_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

#if rank == 0:
#	ut = local_ut
#	for i in range(1, nCPUs):
#		ut_tmp = comm.recv(source = i)
#		ut = np.concatenate((ut, ut_tmp))
#else:
#	comm.send(local_ut, dest = 0)

#ut = comm.bcast(ut, root = 0)
#----------------------

#--------- u ----------
#u_previous = 0.0
#ut_previous = 0.0
#if rank == 0:
#	u += ut * dt

#	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
#	ut_previous = ut.copy()

#u = comm.bcast(u, root=0)
#u_previous = comm.bcast(u_previous, root=0)
#ut_previous = comm.bcast(ut_previous, root=0)
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
	
	epsilon = h.copy()
	
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
	
	#--------- p ----------
	local_P = P_polytrop_mpi(nbeg, nend, rho, T_cld, T_ps, T_0)

	if rank == 0:
		P = local_P
		for i in range(1, nCPUs):
			Ptmp = comm.recv(source = i)
			P = np.concatenate((P, Ptmp))
	else:
		comm.send(local_P, dest = 0)

	P = comm.bcast(P, root = 0)
	#----------------------

	#--------- c ----------
	local_c = sound_speed_mpi(nbeg, nend, rho, T_cld, T_ps, T_0)

	if rank == 0:
		c = local_c
		for i in range(1, nCPUs):
			ctmp = comm.recv(source = i)
			c = np.concatenate((c, ctmp))
	else:
		comm.send(local_c, dest = 0)

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
	
	#------ acc_sph -------
	local_acc_sph = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha, beta, eta)
	
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
	
	#--------- ut ---------
	#local_ut = get_dU_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)
	
	#if rank == 0:
	#	ut = local_ut
	#	for i in range(1, nCPUs):
	#		ut_tmp = comm.recv(source = i)
	#		ut = np.concatenate((ut, ut_tmp))
	#else:
	#	comm.send(local_ut, dest = 0)
	#
	#ut = comm.bcast(ut, root = 0)
	#----------------------

	#--------- u ----------
	#if rank == 0:
	#	u = u_previous + 0.5 * dt * (ut + ut_previous)

	#u = comm.bcast(u, root=0)
	#----------------------
	
	# Heating and Cooling implementation comes here !!!!
	
	#----- u previous -----
	#if rank == 0:
	#	u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
	#	ut_previous = ut.copy()
	
	#u_previous = comm.bcast(u_previous, root=0)
	#ut_previous = comm.bcast(ut_previous, root=0)
	#----------------------
	
	t += dt
	
	if rank == 0:
		if not (ii%50):
			print('h/c = ', np.sort(h/c))

	if rank == 0:
		ii += 1
		dictx = {'pos': r, 'v': v, 'm': m, 'dt': dt, 'current_t': t, 'rho': rho, 'h': h}
		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	
	if rank == 0:
		print('Loop time = ', time.time() - TLoop)

print('elapsed time = ', time.time() - TA)




