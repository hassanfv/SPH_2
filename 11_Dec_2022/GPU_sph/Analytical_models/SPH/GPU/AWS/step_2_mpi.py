
import numpy as np
from mpi4py import MPI
import os
from libsx2_2t import *
import pickle
import time

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

# 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'epsilon': epsilon, 'u': u

with open('tmp_IC.pkl', 'rb') as f:
	data = pickle.load(f)

x = data['x']
y = data['y']
z = data['z']

NG = int(x.shape[0] / 2)
rG = np.hstack((x[:NG].reshape(-1, 1), y[:NG].reshape(-1, 1), z[:NG].reshape(-1, 1)))
print(x.shape, NG, rG.shape)

vx = data['vx']
vy = data['vy']
vz = data['vz']
m = data['m']
epsilon = data['epsilon']
u = data['u']

N = rG.shape[0]

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
	Th2 = time.time()
#--------- h (main) ---------
local_h = smoothing_length_mpi(nbeg, nend, rG)
h = 0.0

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		htmp = comm.recv(source = i)
		h = np.concatenate((h, htmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)
comm.Barrier()
if rank == 0:
	print('Th2 = ', time.time() - Th2)
#----------------------------


if rank == 0:

	epsilon = np.hstack((h, epsilon))

	x = np.round(x, 5)
	y = np.round(y, 5)
	z = np.round(z, 5)

	vx = np.round(vx, 5)
	vy = np.round(vy, 5)
	vz = np.round(vz, 5)

	x = x.astype(np.float32)
	y = y.astype(np.float32)
	z = z.astype(np.float32)

	vx = vx.astype(np.float32)
	vy = vy.astype(np.float32)
	vz = vz.astype(np.float32)

	m = m.astype(np.float32)
	h = h.astype(np.float32)
	epsilon = epsilon.astype(np.float32)
	u = u.astype(np.float32)

	# Save the arrays to a binary file
	N_tot = 2 * NG
	num = str(int(np.floor(N_tot/1000)))
	with open('IC_AGN_EqualN_' + num + 'k.bin', "wb") as file:
	    file.write(x.tobytes())
	    file.write(y.tobytes())
	    file.write(z.tobytes())
	    
	    file.write(vx.tobytes())
	    file.write(vy.tobytes())
	    file.write(vz.tobytes())
	    
	    file.write(m.tobytes())
	    file.write(h.tobytes())
	    file.write(epsilon.tobytes())
	    file.write(u.tobytes())

	os.remove('tmp_IC.pkl')
	
	print()
	print('****************************')
	print('Step_2 Successfully Finished !!!')
	print('****************************')
	#----------------------------



