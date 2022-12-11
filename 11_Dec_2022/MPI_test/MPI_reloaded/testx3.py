

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()

a = 2.0
b = 3.0


def hfunc(x, n):
	return x * n


val = hfunc(a, rank)

if rank == 0:

	res = [val]
	
	for i in range(1, p):
	
		tmp = [comm.recv(source = i)]
	
		res += tmp

else:

	comm.send(val, dest = 0)


if rank == 0:

	res = np.array(res).reshape((1, p))











