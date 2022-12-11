
from mpi4py import MPI
import numpy as np
from numba import njit


@njit
def acc(nbeg, nend, N, P, rho):

	M = nend - nbeg

	ax = np.zeros(M)
	ay = np.zeros(M)
	az = np.zeros(M)

	for i in range(nbeg, nend):
	
		s = 0.0
	
		for j in range(N):

			s += P[i]/rho[i] + P[j]/rho[j]
			
		ax[i-nbeg] = s * 1.0
		ay[i-nbeg] = s * 2.0
		az[i-nbeg] = s * 3.0
	
	ax = ax.reshape((M, 1))
	ay = ay.reshape((M, 1))
	az = az.reshape((M, 1))
	
	a = np.hstack((ax, ay, az))

	return a





comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


N = 10000

gama = 5./3.

dt = 0.1

u = 0.05 + np.zeros(N)
rho = 20.0 + np.zeros(N)
P = (gama - 1.0) * rho * u

r = np.random.random((N, 3))
v = np.random.random((N, 3))

local_N = N / nCPUs

nbeg = rank * local_N
nend = (rank + 1) * local_N

accel = acc(int(nbeg), int(nend), N, P, rho)

if rank == 0:

	Acc = accel
	
	for i in range(1, nCPUs):
	
		accTmp = comm.recv(source = i)
		Acc = np.concatenate((Acc, accTmp))

else:

	comm.send(accel, dest = 0)



if rank == 0:
	
	print(Acc.shape)


for j in range(10):


	if rank == 0:
		
		v += Acc * dt
		r += v * dt
		
	# update h
	# update rho
	# update P
	
	
	accel = acc(int(nbeg), int(nend), N, P, rho)

	if rank == 0:

		Acc = accel
		
		for i in range(1, nCPUs):
		
			accTmp = comm.recv(source = i)
			Acc = np.concatenate((Acc, accTmp))

	else:

		comm.send(accel, dest = 0)


	if rank == 0:
		print('j = ', j)














