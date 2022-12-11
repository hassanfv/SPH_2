
from mpi4py import MPI
import numpy as np
import time
from numba import njit


@njit
def func1(N):

	k = 0
	
	res = np.zeros(N*N)

	for i in range(N):
	
		for j in range(N):
		
			s = j + i
		
			res[k] = s
			
			k += 1
	
	return res



N = 15000


#TA = time.time()
#res = func1(N)
#print('Elapsed time = ', time.time() - TA)
#print(len(res))
#s()


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()


res = func1(N)


if my_rank == 0:

	total = np.array([], dtype = float)
	
	for source in range(1, p):
	
		resT= comm.recv(source = source)
		
		total = np.concatenate((total, resT))

else:

	comm.send(res, dest = 0)


# print the result

if my_rank == 0:

	print('The res is = ', total)
	print('len(res) = ', len(total))



MPI.Finalize




