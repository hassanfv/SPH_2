
# This is in the x-z plane.

import numpy as np
#import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time
from mpi4py import MPI
from libsx import *


#----- read_arrays_from_binary
def read_arrays_from_binary(filename):
    # Read the binary file
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = np.frombuffer(file.read(4), dtype=np.int32)[0]
        NG = np.frombuffer(file.read(4), dtype=np.int32)[0]

        # Read the arrays from the file
        x = np.frombuffer(file.read(N * 4), dtype=np.float32)
        y = np.frombuffer(file.read(N * 4), dtype=np.float32)
        z = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vx = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vy = np.frombuffer(file.read(N * 4), dtype=np.float32)
        vz = np.frombuffer(file.read(N * 4), dtype=np.float32)
        rho = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        h = np.frombuffer(file.read(NG * 4), dtype=np.float32)
        u = np.frombuffer(file.read(NG * 4), dtype=np.float32)

    return x, y, z, vx, vy, vz, rho, h, u, N, NG



#----- densityx
@njit
def densityx(m, WI):
	
	s = 0.
	N = len(m)
	for j in range(N):
	
		s += m[j] * WI[j]
	
	return s



#===== do_smoothingX_single (non-parallel)
@njit
def do_smoothingX_single(r, pos):

	N = pos.shape[0]

	dist = np.zeros(N)
	for j in range(N):

	    dx = pos[j, 0] - r[0]
	    dy = pos[j, 1] - r[1]
	    dz = pos[j, 2] - r[2]
	    dist[j] = (dx**2 + dy**2 + dz**2)**0.5

	hres = np.sort(dist)[50]

	return hres * 0.5



#===== W_I
@njit
def W_I(r, pos, hs, h): # r is the coordinate of a single point. pos contains the coordinates of all SPH particles.

	N = pos.shape[0]
	
	WI = np.zeros(N)

	for j in range(N):

		dx = r[0] - pos[j, 0]
		dy = r[1] - pos[j, 1]
		dz = r[2] - pos[j, 2]
		rr = np.sqrt(dx**2 + dy**2 + dz**2)
		
		hij = 0.5 * (hs + h[j])

		sig = 1.0/np.pi
		q = rr / hij
		
		if q <= 1.0:
			WI[j] = sig / hij**3 * (1.0 - (3.0/2.0)*q**2 + (3.0/4.0)*q**3)

		if (q > 1.0) and (q <= 2.0):
			WI[j] = sig / hij**3 * (1.0/4.0) * (2.0 - q)**3

	return  WI



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


# Specify the input file name
filename = 'G-0.002703.bin'

# Read the arrays from the binary file
x, y, z, vx, vy, vz, rho, h, u, N, NG = read_arrays_from_binary(filename)

xx = np.array(x)
yy = np.array(y)
zz = np.array(z)

pos = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

#plt.scatter(xx, yy, s = 0.1)
#plt.show()

N = NG
m = 1.0 / N + np.zeros(N)

xxyy = 0.2
#x = [-xxyy, xxyy]
x = [-0.04, 0.04]
y = [-xxyy, xxyy]
y = z = [-xxyy, xxyy]

dx = dy = dz = 0.002

xarr = np.arange(x[0]-dx, x[1], dx)
yarr = zarr = np.arange(y[0]-dy, y[1], dy)

print(len(xarr) * len(yarr) * len(zarr))


Nx = len(yarr)
#------- used in MPI --------
count = Nx // nCPUs
remainder = Nx % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#----------------------------


@njit
def get_rho_mpi(nbeg, nend, xarr, yarr, zarr, pos, h):

	M = nend - nbeg
	N = len(yarr)

	rho = np.zeros((M, N))

	for i in range(nbeg, nend):

		for j in range(len(zarr)):
			
			s = 0.
			for k in range(len(xarr)):
				
				r = np.array([xarr[k], yarr[i], zarr[j]])
				hs = do_smoothingX_single(r, pos)
				
				WI = W_I(r, pos, hs, h)
				
				s += densityx(m, WI)
			
			rho[i-nbeg, j] = s
		
	return rho



#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = get_rho_mpi(nbeg, nend, xarr, yarr, zarr, pos, h)
rho = 0.0 

if rank == 0:
	rho = local_rho
	for i in range(1, nCPUs):

		rhotmp = comm.recv(source = i)
		rho = np.concatenate((rho, rhotmp))
else:
	comm.send(local_rho, dest = 0)

rho = comm.bcast(rho, root = 0)
	
if rank == 0:
	print('Trho = ', time.time() - Trho)
#----------------------

dictx = {'rho': rho, 'dx': dx}

if rank == 0:
	with open('Nxy.pkl', 'wb') as f:
		pickle.dump(dictx, f)






