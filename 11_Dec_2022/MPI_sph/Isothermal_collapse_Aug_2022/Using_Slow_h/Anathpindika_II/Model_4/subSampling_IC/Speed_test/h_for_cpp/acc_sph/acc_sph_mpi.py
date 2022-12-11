
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
divV = data['divV']
curlV = data['curlV']
alpha = data['alpha']




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
	TT = time.time()
	
#------ acc_sph mu_visc -------
acc_sph = 0.0
#dt_cv = 0.0
tmp = getAcc_sph_shear_mpi(nbeg, nend, r, v, rho, P, c, h, m, divV, curlV, alpha)

local_acc_sph = tmp
#local_dt_cv = tmp[1]


if rank == 0:
	acc_sph = local_acc_sph
	#dt_cv = local_dt_cv
	for i in range(1, nCPUs):
		tmpt = comm.recv(source = i)
		acc_sph_tmp = tmpt
		#dt_cv_tmp = tmpt[1]
		acc_sph = np.concatenate((acc_sph, acc_sph_tmp))
		#dt_cv = np.concatenate((dt_cv, dt_cv_tmp))
else:
	comm.send(local_acc_sph, dest = 0)

acc_sph = comm.bcast(acc_sph, root = 0)
#dt_cv = comm.bcast(dt_cv, root = 0)
#dt_cv = np.min(dt_cv)
#----------------------

if rank == 0:
	print('TT = ', time.time() - TT)

if rank == 0:
	ax = acc_sph[:, 0]
	ay = acc_sph[:, 1]
	az = acc_sph[:, 2]

	dictx = {'ax': ax, 'ay': ay, 'az': az}
	df = pd.DataFrame(dictx)
	df.to_csv('acc_sph_cpu.csv', index = False)
	
	print(df)






