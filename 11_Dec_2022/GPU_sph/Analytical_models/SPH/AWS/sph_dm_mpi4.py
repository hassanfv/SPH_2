
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
eps_AGN = 0.05
#---------------------------
t = 0.0
dt = 1e-4
tEnd = 3.0
Nt = int(np.ceil(tEnd/dt)+1)


filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('IC.pkl', 'rb') as f:
    data = pickle.load(f)

rG = data['r_gas']
rD = data['r_dm']
vG = data['v_gas']
vD = data['v_dm']
m = data['m']
mBH = data['mBH']
u  = data['u']
epsD = data['eps_dm'] # epsilon (i.e. gravitational softening length) for DM. For gas it will be set equal to the smoothing length h!
h = data['h']
L_Edd = data['L_Edd'] # in energy per unit mass per unit time similar to u !
epsG = h.copy()

epsilon = np.hstack((epsG, epsD))

print('The file is read .....')
print()

NG = rG.shape[0]
ND = rD.shape[0]

r = np.vstack((rG, rD))

v = np.vstack((vG, vD))

# Inserting a Black Hole at the center
rBH = np.array([[0, 0, 0]])
r = np.vstack((r, rBH))
vBH = np.array([[0, 0, 0]])
v = np.vstack((v, vBH))
m = np.hstack((m, np.array([mBH])))
print(m.shape)
print(m)
print(mBH)
epsilon = np.hstack((epsilon, np.array(epsilon[-1])))

hBH = smoothing_BH(rBH, r[:NG])
weights = InjectEnergy_weights(rBH, r[:NG], hBH, h)
weights /= np.sum(weights)
print('hBH = ', hBH)
#-------------------------------------

N = r.shape[0]

#------- used in MPI For gravity calculation so Gas + DM --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#----------------------------


#------- used in MPI For Gas Only --------
count = NG // nCPUs
remainder = NG % nCPUs

if rank < remainder:
	nbegG = rank * (count + 1)
	nendG = nbegG + count + 1
else:
	nbegG = rank * count + remainder
	nendG = nbegG + count
#----------------------------



#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = getDensity_mpi(nbegG, nendG, r[:NG], m[:NG], h)
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

#--------- P ----------
P = 0.0
if rank == 0:
	P = getPressure(rho, u, gama)

P = comm.bcast(P, root = 0)
#----------------------

#--------- c ----------
c = 0.0
if rank == 0:
	c = np.sqrt(gama * (gama - 1.0) * u)

c = comm.bcast(c, root = 0)
#----------------------

#----- div_curlV ------
divV = 0.0
curlV = 0.0
local_div_curlV = div_curlVel_mpi(nbegG, nendG, r[:NG], v[:NG], rho, m[:NG], h)

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
local_acc_sph = getAcc_sph_shear_mpi(nbegG, nendG, r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)

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
	acc = acc_g

acc = comm.bcast(acc, root = 0)
#----------------------

#-------- acc adding sph contribution ---------
if rank == 0:
	acc[:NG] += acc_sph

acc = comm.bcast(acc, root = 0)
#----------------------

#--------- ut ---------
ut = 0.0
local_ut = get_dU_shear_mpi(nbegG, nendG, r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)

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
	u += eps_AGN * L_Edd * dt * weights # Injecting AGN energy! BH

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
		r[-1, :] = [0, 0, 0] # BH position must not be updated! We do not move the BH!
	
	r = comm.bcast(r, root = 0)
	#----------------------

	#--------- h ----------
	local_h = h_smooth_fast_mpi(nbegG, nendG, r[:NG], h)
	
	if rank == 0:
		h = local_h
		for i in range(1, nCPUs):
			htmp = comm.recv(source = i)
			h = np.concatenate((h, htmp))
	else:
		comm.send(local_h, dest = 0)

	h = comm.bcast(h, root = 0)
	#----------------------
	
	#--------- hBH --------
	if rank == 0:
	
		hBH = smoothing_BH(rBH, r[:NG])
		
	hBH = comm.bcast(hBH, root = 0)
	#----------------------
	
	#--------- weights --------
	if rank == 0:

		weights = InjectEnergy_weights(rBH, r[:NG], hBH, h)
		weights /= np.sum(weights)
		
	weights = comm.bcast(weights, root = 0)
	#----------------------
	
	#-------- rho ---------
	local_rho = getDensity_mpi(nbegG, nendG, r[:NG], m[:NG], h)
	
	if rank == 0:
		rho = local_rho
		for i in range(1, nCPUs):
			rhotmp = comm.recv(source = i)
			rho = np.concatenate((rho, rhotmp))
	else:
		comm.send(local_rho, dest = 0)
	
	rho = comm.bcast(rho, root = 0)	
	#----------------------
	
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
		P = getPressure(rho, u, gama)
	
	P = comm.bcast(P, root = 0)
	#----------------------

	#--------- c ----------
	if rank == 0:
		c = np.sqrt(gama * (gama - 1.0) * u)
	
	c = comm.bcast(c, root = 0)
	#----------------------
	
	#----- div_curlV ------
	local_div_curlV = div_curlVel_mpi(nbegG, nendG, r[:NG], v[:NG], rho, m[:NG], h)

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
	local_acc_sph = getAcc_sph_shear_mpi(nbegG, nendG, r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
	
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
		acc = acc_g
	
	acc = comm.bcast(acc, root = 0)
	#----------------------
	
	#-------- acc adding sph contribution ---------
	if rank == 0:
		acc[:NG] += acc_sph

	acc = comm.bcast(acc, root = 0)
	#----------------------
	
	#--------- v ----------
	if rank == 0:
		v += acc * dt/2.0
	
	v = comm.bcast(v, root = 0)
	#----------------------
	
	#--------- ut ---------
	local_ut = get_dU_shear_mpi(nbegG, nendG, r[:NG], v[:NG], rho, P, c, h, m[:NG], divV, curlV, alpha)
	
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
	if rank == 0:
		u = u_previous + 0.5 * dt * (ut + ut_previous)
		u += eps_AGN * L_Edd * dt * weights # Injecting AGN energy! BH

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
		if not (ii%10):
			print('h/c = ', np.sort(h/c))

	if rank == 0:
		ii += 1
		dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho, 'NG': NG, 'ND': ND}
		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	
	if rank == 0:
		print('Loop time = ', time.time() - TLoop)

print('elapsed time = ', time.time() - TA)




