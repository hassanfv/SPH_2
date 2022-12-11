
# The difference with _v3 is that here the particles can have different mass and the gravitational acceleration calculation is accordingly modified.
# The difference with _v2 is that here we incorporate shear viscosity by using the Balsara switch.
# The difference with previous version is that here we separated u and u_previous, ut_previous updates separately. See below.
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx2_2t import *
from mpi4py import MPI
import pandas as pd

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


namx = 'GPU_IC_Anathpin_RAND_tmp.csv'
df = pd.read_csv(namx)

x = df['x'].values
y = df['y'].values
z = df['z'].values

r = np.array([x, y, z])
r = r.T
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



#----- using previous step as the hprevious.
h_previous = h.copy()
local_h = smoothing_fast_new_mpi2(nbeg, nend, r, h_previous)
h = 0.0

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		htmp = comm.recv(source = i)
		h = np.concatenate((h, htmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)




epsilon = 0.0001

if rank == 0:
	df['h'] = np.round(h, 6)
	num = str(int(len(h)/2/1000))
	nam = namx[:-13] + '_' + num + 'k_RAND' + '.csv'
	df['eps'] = epsilon
	df.to_csv(nam, index = False, header = False)
	os.remove(namx)







