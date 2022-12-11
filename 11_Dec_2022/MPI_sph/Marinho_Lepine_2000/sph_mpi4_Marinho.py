
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx import *
from mpi4py import MPI


#============= COOLING FROM TABLE ====================
with open('./CoolingGrids_project/u_after_cool_MPI_dt_0.005_N_200.pkl', 'rb') as f:
	data = pickle.load(f)

NN = int(np.sqrt(data.shape[0]))

rhoG = data[:, 0].reshape(NN, NN)
uBefore = data[:, 1].reshape(NN, NN)
uAfter = data[:, 2].reshape(NN, NN)
dt = data[:, 3].reshape(NN, NN)

u_uniq = np.unique(uBefore)
rho_uniq = np.unique(rhoG)


@njit
def do_grid_cooling(ut, rhot):

	N = len(ut)
	ures = np.zeros(N)

	for i in range(N):

		#-------- n11 i.e. lower left -----------
		n11 = np.where(u_uniq <= ut[i])[0]
		u11 = np.max(u_uniq[n11])

		n11 = np.where(rho_uniq <= rhot[i])[0]
		rho11 = np.max(rho_uniq[n11])
		#----------------------------------------

		ndx = np.where((uBefore == u11) & (rhoG == rho11))

		nrow = ndx[0][0]
		ncol = ndx[1][0]

		ures[i] = uAfter[nrow, ncol]

	return ures
#=====================================================



@njit
def do_grid_cooling_mpi(nbeg, nend, ut, rhot):
	
	M = nend - nbeg
	ures = np.zeros(M)

	for i in range(nbeg, nend):

		#-------- n11 i.e. lower left -----------
		n11 = np.where(u_uniq <= ut[i])[0]
		u11 = np.max(u_uniq[n11])

		n11 = np.where(rho_uniq <= rhot[i])[0]
		rho11 = np.max(rho_uniq[n11])
		#----------------------------------------

		ndx = np.where((uBefore == u11) & (rhoG == rho11))

		nrow = ndx[0][0]
		ncol = ndx[1][0]

		ures[i-nbeg] = uAfter[nrow, ncol]

	return ures
#=====================================================


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
dt = 0.005
tEnd = 5.0
Nt = int(np.ceil(tEnd/dt)+1)

filz = np.sort(os.listdir('./Outputs'))
try:
	for k in range(len(filz)):
		os.remove('./Outputs/' + filz[k])
except:
	pass


with open('Marinho_IC_5000.pkl', 'rb') as f:
    res = pickle.load(f)

r = res['r']
v = res['v']

print('The file is read .....')
print()

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

epsilon = np.zeros(N) + 0.02

MSPH = 10.0 # total gas mass


uFloor = 0.422 # 0.422 corresponds to T = 20 K. I used u_as_a_func_of_T function to calculate it !
u = np.zeros(N) + uFloor

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
ucool = 0.0 # similar to rho = 0.0 above !

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

#------- acc_g --------
if rank == 0:
	TG = time.time()
	
local_acc_g = getAcc_g_smth_mpi(nbeg, nend, r, m, G, epsilon)
acc_g = 0.0

if rank == 0:
	acc_g = local_acc_g
	for i in range(1, nCPUs):
		acc_gtmp = comm.recv(source = i)
		acc_g = np.concatenate((acc_g, acc_gtmp))
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

#------ acc_sph -------
acc_sph = 0.0
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
acc = 0.0
if rank == 0:
	acc = acc_g + acc_sph

acc = comm.bcast(acc, root = 0)
#----------------------

#--------- ut ---------
ut = 0.0
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
	
	
	
	#------ Setting epsilon = h -------
	if rank == 0:
		epsilon = h.copy()
	
	epsilon = comm.bcast(epsilon, root = 0)
	#----------------------------------
	
	
	
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
	if rank == 0:
		u = u_previous + 0.5 * dt * (ut + ut_previous)

	u = comm.bcast(u, root=0)
	#----------------------

	#------ cooling -------
	local_u_cool = do_grid_cooling_mpi(nbeg, nend, u, rho)
	
	if rank == 0:
		ucool = local_u_cool
		for i in range(1, nCPUs):
			ucooltmp = comm.recv(source = i)
			ucool = np.concatenate((ucool, ucooltmp))
	else:
		comm.send(local_u_cool, dest = 0)

	u = comm.bcast(ucool, root = 0)
	#----------------------
	
	#------ u floor -------
	if rank == 0:
		u[u < 0.12] = 0.12
	u = comm.bcast(u, root=0)
	#----------------------
	
	if rank == 0:
		print('u before cooling = ', u[1], u[100], u[1232], u[2634], u[3763], np.median(u))
	

	#----- u_previous -----
	if rank == 0:
		u_previous = u.copy() # since u_previous and ut_previous is only used in rank = 0, we do not need to broadcast them.
		ut_previous = ut.copy()
	
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
		dictx = {'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho}
		with open('./Outputs/' + str(ii).zfill(5) + '.pkl', 'wb') as f:
			pickle.dump(dictx, f)
	
	if rank == 0:
		print('Loop time = ', time.time() - TLoop)

print('elapsed time = ', time.time() - TA)




