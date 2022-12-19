
import numpy as np
from photolibs2 import *
import time
import pickle
from mpi4py import MPI
from numba import njit

def cooler(nbeg, nend):

	M = nend - nbeg
	N = len(uGrid)
	res = np.zeros((M*N, 4))
	k = 0

	# Each loop takes 0.133 seconds!
	for i in range(nbeg, nend):

		for j in range(len(uGrid)):
			
			ux = DoCooling_h(rhoGrid[i], uGrid[j], dt_t, XH)
			delta_u = uGrid[j] - ux
			
			res[k, 0] = uGrid[j] # u_ad ====> u before cooling
			res[k, 1] = rhoGrid[i]
			res[k, 2] = dt_t
			res[k, 3]= delta_u
			
			k+=1
	
	return res


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

XH = 0.76
mH = 1.6726e-24 # gram

N = N_rho = 1000
N_u = 1000

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

Tmin = 1e4
Tmax = 1e6
Tgrid = np.logspace(np.log10(Tmin), np.log10(Tmax), N_u)
# converting T to u
nHcgs = 1.0 # cm^-1. This value is not very important. We just want to have a grid for u !! You could put nHcgs = 0.1, or 0.01, or ... !!!
uGrid = np.array([convert_Temp_to_u(T, nHcgs, XH) for T in Tgrid])

nH_min = 1e-4
nH_max = 1e3
rho_min = nH_min * mH
rho_max = nH_max * mH
rhoGrid = np.logspace(np.log10(rho_min), np.log10(rho_max), N_rho)

dt_t  = 500 * 3600. * 24. * 365.24 # 500 YEARS

res = np.zeros((N_u*N_rho, 4))

print('res shape = ', res.shape)
print(res.nbytes/1024/1024) # in Mb


if rank == 0:
	TT = time.time()

TA = time.time()
#--------- cooler (main) ---------
local_res = cooler(nbeg, nend)
res = 0

if rank == 0:
	res = local_res
	for i in range(1, nCPUs):
		res_tmp = comm.recv(source = i)
		res = np.concatenate((res, res_tmp))
else:
	comm.send(local_res, dest = 0)

res = comm.bcast(res, root = 0)
comm.Barrier()
#if rank == 0:
print('TA = ', time.time() - TA)
#----------------------------

if rank == 0:
	print('Total Elapsed time = ', time.time() - TT)

if rank == 0:
	print(res)
	with open('coolingGrid.pkl', 'wb') as f:
		pickle.dump(res, f)



