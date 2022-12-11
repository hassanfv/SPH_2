
import numpy as np
import time
import pickle
import os
from libsx import *
from mpi4py import MPI
from shear_test import *
import matplotlib.pyplot as plt

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


with open('ICM.pkl', 'rb') as f:
    data = pickle.load(f)

r = data['r'] # in code unit.
m = data['m'] # in code unit.

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
	Th1 = time.time()
#----------- h ---------------
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
#----------------------------
if rank == 0:
	print('Th1 = ', time.time() - Th1)


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


if rank == 0:

	print()
	print('******************************')
	print('median(rho) = ', np.median(rho))
	print('******************************')



