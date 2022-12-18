
import numpy as np
from mpi4py import MPI
import os
from libsx2_2t import *
import pickle
import time

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

with open('tmp1.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
v = data['v']
m = data['m']
paramz = data['paramz']
u = data['u']

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
	print('rho = ', np.sort(rho))

	dictx = {'r': r, 'v': v, 'm': m, 'u': u, 'h': h, 'rho': rho, 'paramz': paramz}
	
	num = str(int(np.floor(len(m)/1000)))
	
	with open('Main_IC_Grid_' + str(num) + 'k.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	os.remove('tmp1.pkl')
	
	print()
	print('****************************')
	print('Step_2 Successfully Done !!!')
	print('****************************')
	#----------------------------


