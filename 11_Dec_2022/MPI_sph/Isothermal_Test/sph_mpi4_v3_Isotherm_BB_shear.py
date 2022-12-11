
# the difference with _v2 is that here we incorporate shear viscosity by using the Balsara switch.
# difference with previous version is that here we separated u and u_previous, ut_previous updates separately. See below.
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx import *
from mpi4py import MPI
from shear import *


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

M_sun = 1.989e33 # gram
G_cgs = 6.67259e-8 #  cm3 g-1 s-2
R = 3.2e16 # cm
M = 1. * M_sun
#---- For Unit ref. see Thacker et al - 2000 ------
unitVelocity = (G_cgs * M / R)**0.5
#unitDensity = 3.*M/4./np.pi/R**3
unitDensity = M/R**3
unit_u = G_cgs*M/R
unitTime = (R**3/G_cgs/M)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6
unit_P = unitDensity * unit_u

mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

mu_bar = 2.29 # For the Boss & Bodenheimer test. See: units.F90 at https://github.com/dhubber/seren/tree/master/src/setup

Pconst = kB / mu_bar / mH
T_0 = 10. # K
rho_crit = 1e-14

#---- Constants -----------
eta = 0.1
gamma = 5.0/3.0
alpha = 1.0
beta = 2.0
G = 1.0
#---------------------------
t = 0.0
dt = 0.001
tEnd = 5.0
Nt = int(np.ceil(tEnd/dt)+1)

unitVelocity =  64400.55684444817 # see IC_isothermal.py code
R_gas_in_cgs = 8.314e7 # erg/mol/K
T_gas_in_K = 10. # K


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

with open('Boss_IC_3000.pkl', 'rb') as f:
	dict_r_v = pickle.load(f)

r = dict_r_v['r']
v = dict_r_v['v']
#m = dict_r_v_m['m']

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

epsilon = np.zeros(N) + 0.10

MSPH = 1.0 # total gas mass

uFloor = 0.15 # corresponding to T = 10 K for a gas of pure molecular hydrogen
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

if rank == 0:
	Th1 = time.time()
#-------- h (initial) -------
local_h = smoothing_length_mpi(nbeg, nend, r)  # This plays the role of the initial h so that the code can start !
h = 0 # This is just a placeholder. Its absence would crash this line :h = comm.bcast(h, root = 0) for CPUs other than rank = 0.

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		h_tmp = comm.recv(source = i)
		h = np.concatenate((h, h_tmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)
#----------------------------
if rank == 0:
	print('Th1 = ', time.time() - Th1)


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



#---- P Barotropic ----
Temp = 0.0
P = 0.0
if rank == 0:
	Temp = T_0 * (1. + rho*unitDensity/rho_crit)**(gamma-1.)
	P = Pconst * rho*unitDensity * Temp / unit_P

P = comm.bcast(P, root = 0)
#----------------------

#--------- c ---------- Isothermal sound speed ! # ==>NOW Ideal EOS is active !!!!!!!!!!!!!!
c = 0.0
if rank == 0:
	c = np.sqrt(P/rho)

c = comm.bcast(c, root = 0)
#----------------------

#--------- P ---------- Isothermal eq. of state ! # ==>NOW Ideal EOS is active !!!!!!!!!!!!!!
#P = 0.0
#if rank == 0:
	#P = getPressure(rho, u, gamma)
#	P = c*c * rho

#P = comm.bcast(P, root = 0)
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
#local_acc_sph = getAcc_sph_mpiZ(nbeg, nend, r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
local_acc_sph = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

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

if rank == 0:
	#------ Jeans Mass from Hubber et al - 2006 -------
	M_Jeans = np.pi**2.5 * c**3 / 6. / G**1.5 / rho**0.5
	#--------------------------------------------------
	print('M_Jeans = ', np.sort(M_Jeans))

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




	#---- P Barotropic ----
	Temp = 0.0
	P = 0.0
	if rank == 0:
		Temp = T_0 * (1. + rho*unitDensity/rho_crit)**(gamma-1.)
		P = Pconst * rho*unitDensity * Temp / unit_P

	P = comm.bcast(P, root = 0)
	#----------------------

	#--------- c ---------- Isothermal sound speed ! # ==>NOW Ideal EOS is active !!!!!!!!!!!!!!
	c = 0.0
	if rank == 0:
		c = np.sqrt(P/rho)

	c = comm.bcast(c, root = 0)
	#----------------------



	#--------- c ---------- Isothermal ==>NOW Ideal EOS is active !!!!!!!!!!!!!!
	#if rank == 0:
		#c = np.sqrt(gama * (gama - 1.0) * u)
	#	c = np.zeros(N) + np.sqrt(R_gas_in_cgs*T_gas_in_K) / unitVelocity
	
	#c = comm.bcast(c, root = 0)
	#----------------------
	
	#--------- P ---------- Isothermal eq. of state ! # ==>NOW Ideal EOS is active !!!!!!!!!!!!!!
	#if rank == 0:
		#P = getPressure(rho, u, gama) 
	#	P = c*c * rho
	
	#P = comm.bcast(P, root = 0)
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
	#local_acc_sph = getAcc_sph_mpiZ(nbeg, nend, r, v, rho, P, c, h, m, gamma, eta, alpha, beta)
	local_acc_sph = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)
	
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
	
	if rank == 0:
		#------ Jeans Mass from Hubber et al - 2006 -------
		M_Jeans = np.pi**2.5 * c**3 / 6. / G**1.5 / rho**0.5
		#--------------------------------------------------
		print('M_Jeans = ', np.sort(M_Jeans))
	
	#--------- ut ---------
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
		dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	
	if rank == 0:
		print('Loop time = ', time.time() - TLoop)

print('elapsed time = ', time.time() - TA)




