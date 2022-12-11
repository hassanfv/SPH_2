
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
from shear_test3_t import *
import pandas as pd


np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


with open('hydroData_130k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

v = data['v']
vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

rho = data['rho']
P = data['P']
c = data['c']
h = data['h']
m = data['m']
#divV = data['divV']
#curlV = data['curlV']
alpha = data['alpha']


N = r.shape[0]

if rank == 0:
	print(r.shape)
	print()
	print('m = ', m)
	print()
	print('v = ', v)
	print()

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


#----- div_curlV ------

if rank == 0:
	TT = time.time()

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

if rank == 0:
	print('TT = ', time.time() - TT)
#----------------------

dictx = {'divV': divV, 'curlV': curlV}

df = pd.DataFrame(dictx)

df.to_csv('div_curlV.csv', index = False)

if rank == 0:
	print(df)





