
import numpy as np
import time
import pickle
import os
from libsx2 import *
from mpi4py import MPI
from shear_test2 import *


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


with open('00012.pkl', 'rb') as f:
	data = pickle.load(f)


r = data['pos']
h = data['h']

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

#--------- h ----------
if rank == 0:
	T_h = time.time()

#h_previous = h.copy()
#local_h = smoothing_length_mpix(nbeg, nend, r, h_previous)
local_h = smoothing_length_mpi(nbeg, nend, r)

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		htmp = comm.recv(source = i)
		h = np.concatenate((h, htmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)

if rank == 0:
	print()
	print('T_h = ', time.time() - T_h)
	print()
#----------------------

if rank == 0:
	print('h = ', np.sort(h))




