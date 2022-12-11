
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import time
from mpi4py import MPI
import os
from libsx2_2t import *


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


with open('main_IC_tmp1.pkl', 'rb') as f:
	data = pickle.load(f)

r = res = data['r']
rho_0 = data['rho_cen']
c_0 = data['c_0']
gamma = data['gamma']
Rcld_in_cm = data['Rcld_in_cm']
Rcld_in_pc = data['Rcld_in_pc']
Mcld_in_g = data['Mcld_in_g']
muu = data['mu']
grav_const_in_cgs = data['grav_const_in_cgs']


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

if rank == 0:
	Th2 = time.time()
#--------- h (main) ---------
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


m = np.ones(N) / N

#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = getDensity_mpi(nbeg, nend, r, m, h)
rho = 0.0

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



if rank == 0:
	print()
	print(np.sort(rho))

	dictx = {'r': res, 'h': h, 'rho': rho, 'rho_cen': rho_0, 'c_0': c_0, 'gamma': gamma, 'Rcld_in_pc': Rcld_in_pc, 
	 'Rcld_in_cm': Rcld_in_cm, 'Mcld_in_g': Mcld_in_g, 'mu': muu, 'grav_const_in_cgs': grav_const_in_cgs}

	with open('main_IC_tmp2.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	os.remove('main_IC_tmp1.pkl')
	#----------------------------






