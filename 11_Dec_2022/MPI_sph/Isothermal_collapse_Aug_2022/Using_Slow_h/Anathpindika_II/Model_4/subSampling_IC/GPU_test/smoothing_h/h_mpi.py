
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
from shear_test3_t_del import *
import pandas as pd


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

df = pd.read_csv('../test_data.csv', names = ['x', 'y', 'z', 'hprevious'])
x = df['x'].values
y = df['y'].values
z = df['z'].values

r = np.array([x, y, z]).T

print(r.shape)

h_previous = df['hprevious'].values


N = x.shape[0]

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

if rank ==0:
	T_h = time.time()

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

if rank == 0:
	print()
	print('T_h = ', time.time() - T_h)
	print()
	for i in range(10):
		print(np.round(h_previous[i], 4), np.round(h[i], 4))
#----------------------



