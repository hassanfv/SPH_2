
from mpi4py import MPI
import numpy as np
from numba import njit



#*** cfunc
def cfunc(a, b):

	N = len(a)
	c = np.zeros(N)

	for i in range(N):
	
		c[i] = a[i] + b[i]

	return c



#*** afunc
def afunc(b, c):

	N = len(a)
	a = np.zeros(N)

	for i in range(N):
	
		a[i] = b[i] + c[i]

	return a



#*** afunc
def bfunc(a, c):

	N = len(a)
	b = np.zeros(N)

	for i in range(len(a)):
	
		b[i] = a[i] + c[i]

	return a





comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


a = np.random.random(1000)
b = np.random.random(1000)
c = np.random.random(1000)

N = 20000
Nstp = N / nCPUs

nstart = int(rank * Nstp)
nstop = int(nstart + Nstp)


local_a = a[nstart:nstop]
local_b = b[nstart:nstop]
local_c = cfunc(local_a, local_b)


if rank == 0:

	c = local_c * 100
	
	for i in range(1, int(nCPUs)):
	
		cct = comm.recv(source = i)
		c = np.concatenate((c, cct))
		print('****')

else:

	comm.send(local_c, dest = 0)


c = comm.bcast(c, root=0)

local_c = cfunc(local_a, local_b)


local_b = bfunc(local_a, local_c)


#if rank == 0:
#	print(np.sort(c))



#local_a = afunc(local_b, local_c)

#local_b = bfunc(local_a, local_c)







