
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



filename = 'hfv_IC_RND_8k_tmp.pkl'

with open(filename, 'rb') as f:
	data = pickle.load(f)

print('The file is read .....')
print()

r = data['r']
v = data['v'] #/ unitVelocity
#h = data['h']
m = data['m']
t_ff = data['t_ff']

minimum_h = 0.0001 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

unitVelocity = data['unitVelocity']
unitTime = data['unitTime']
print('unitVelocity (See hfv_IC_generator) = ', unitVelocity)
print('unitTime (See hfv_IC_generator) = ', unitTime)

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
#local_h = h_smooth_fast_mpi(nbeg, nend, r, h)
local_h = h_smooth_fast_mpi_min_h_set(nbeg, nend, r, h, minimum_h)

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


if rank == 0:

	dictx = {'r': r, 'v': v, 'h': h, 'm': m,
		 'unitVelocity': unitVelocity,
		 'unitTime': unitTime,
		 't_ff': t_ff}

	num = str(int(np.floor(r.shape[0]/1000)))

	with open('hfv_IC_RND_' + num +'k.pkl', 'wb') as f:
	    pickle.dump(dictx, f)
	
	os.remove(filename)






