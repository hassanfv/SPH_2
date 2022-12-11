
from mpi4py import MPI
import numpy as np
import time
from numba import njit
import pickle
import get_dU as gd




with open('Evrard_7208.pkl', 'rb') as f:   # !!!!!! Change epsilon
    res = pickle.load(f)
resx = res['x'].reshape((len(res['x']),1))
resy = res['y'].reshape((len(res['x']),1))
resz = res['z'].reshape((len(res['x']),1))

pos = np.hstack((resx, resy, resz))
N = pos.shape[0]
v = np.zeros_like(pos) + 0.001
h = 0.02 + np.zeros(N)
m = 1.0/1472. + np.zeros(N)
gama = 5/3.
eta = 0.01
alpha = 1.0
beta = 2.0

rho = np.zeros(N) + 27.0
u = 0.05 + np.zeros(N)

P = (gama - 1.) * rho * u
c = np.sqrt(gama * (gama - 1.0) * u)

ut = gd.get_dU(pos, v, rho, P, c, h, m, gama, eta, alpha, beta)


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()



if my_rank == 0:

	total = ut
	
	for source in range(1, p):
	
		resT= comm.recv(source = source)
		
		total = np.concatenate((total, resT))

else:

	comm.send(ut, dest = 0)



# print the result
if my_rank == 0:

	print('The res shape = ', total.shape)



MPI.Finalize




