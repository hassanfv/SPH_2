
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time
from mpi4py import MPI


#===== smoothing_length_mpi (same as do_smoothingX but modified fpr MPI)
@njit
def smoothing_length_mpi(nbeg, nend, pos):

	N = pos.shape[0]
	M = nend - nbeg
	hres = np.zeros(M)

	for i in range(nbeg, nend):

		dist = np.zeros(N)

		for j in range(N):

		    dx = pos[j, 0] - pos[i, 0]
		    dy = pos[j, 1] - pos[i, 1]
		    dz = pos[j, 2] - pos[i, 2]
		    dist[j] = (dx**2 + dy**2 + dz**2)**0.5

		hres[i-nbeg] = np.sort(dist)[50]

	return hres * 0.5




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


#filez = glob.glob('./Outputs_/*.pkl')
filez = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_Model_3_Anathpindika_2009/*.pkl'))

j = 370

with open(filez[j], 'rb') as f:
	data = pickle.load(f)

pos = data['pos']

N = pos.shape[0]
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


#------------ h -------------
if rank == 0:
	Thh = time.time()

local_h = smoothing_length_mpi(nbeg, nend, pos)
h = 0 # This is just a placeholder. Its absence would crash this line :h = comm.bcast(h, root = 0) for CPUs other than rank = 0.

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		h_tmp = comm.recv(source = i)
		h = np.concatenate((h, h_tmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)

if rank == 0:
	print('Thh = ', time.time() - Thh)
#----------------------------


if rank == 0:
	with open('hh.pkl', 'wb') as f:
		pickle.dump(h, f)





