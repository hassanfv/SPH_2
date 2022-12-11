
from mpi4py import MPI
import numpy as np
import time
from numba import njit


def f(x):

	return x*x


@njit
def Trap(a, b, n, h):

	integral = (a*a + b*b) / 2.0
	
	x = a
	
	for i in range(1, int(n)):
	
		x = x + h
		
		integral += x*x
		
	return integral*h



#TA = time.time()
#print(Trap(0., 1., 40000000000, 1./40000000000))
#print('Elapsed time = ', time.time() - TA)
#s()


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

a = 0.0
b = 1.0
n = 40000000000

dest = 0
total = -1 # -1 is just a place holder.

h = (b - a)/n
local_n = n / p

local_a = a + my_rank * local_n * h
local_b = local_a + local_n * h

integral = Trap(local_a, local_b, local_n, h)


if my_rank == 0:

	total = integral
	
	for source in range(1, p):
	
		integral = comm.recv(source = source)
		
		total += integral

else:

	comm.send(integral, dest = 0)


# print the result

if my_rank == 0:

	print('The integral is = ', total)



MPI.Finalize




