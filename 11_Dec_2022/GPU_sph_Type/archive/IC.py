
import numpy as np
#import matplotlib.pyplot as plt
import time
from libsx import *
#import pandas as pd
import pickle
from mpi4py import MPI

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

if rank == 0:
	TT = time.time()

N_particles = 10000 # desired number of SPH particles in the simulation.

f_gas = 0.17
NGas = int(f_gas * N_particles)
NDM = N_particles - NGas

# Create a 3D grid of particles
L_box_in_kpc = 1.0
r = np.random.uniform(0, L_box_in_kpc, (N_particles, 3)) # The unit of length is 1 kpc!

r = r - L_box_in_kpc / 2.0 # Since unit_length is 1 kpc therefore this r is already in unit length!

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

v = np.random.uniform(0, 5.0, (N_particles, 3)) #np.zeros_like(r)

vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

mass = np.full(N_particles, 1. / N_particles)
if rank == 0:
	print('mass = ', mass)

u_0 = 0.01
u_tmp = np.full(NGas, u_0)

u = np.zeros(N_particles)
u[:NGas] = u_tmp


#plt.scatter(r[:, 0], r[:, 1], s = 0.01, color = 'k')
#plt.show()

N = NGas

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
    print('r.shape = ', r.shape)
    Th2 = time.time()
#--------- h (main) ---------
local_h = smoothing_length_mpi(nbeg, nend, r[:NGas, :])
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

htmp = h.copy()

h = np.zeros(N_particles)
h[:NGas] = htmp

if rank == 0:
	print('h = ', h)
	print('h.shape = ', h.shape)
	print('r.shape = ', r.shape)

if rank == 0:
	epsilon = np.full(N_particles, np.median(h))

	#*************************************************************************
	#*********** Updating the arrays for AGN outflow injection ***************
	#*************************************************************************

	# We will extend the arrays by N_blank = 10000, which means maximally 10000 outflow particles can be 
	# injected without any issue. If we think it may exceed this value, we have to adjust this
	# value!

	N_blank = 1000
	blank = np.zeros(N_blank)
	
	print(x.shape)
	
	x = np.hstack((x, blank))
	y = np.hstack((y, blank))
	z = np.hstack((z, blank))
	
	vx = np.hstack((vx, blank))
	vy = np.hstack((vy, blank))
	vz = np.hstack((vz, blank))
	
	mass = np.hstack((mass, blank))
	u = np.hstack((u, blank))
	h = np.hstack((h, blank))
	epsilon = np.hstack((epsilon, blank))
	
	Typ = np.full(len(x), -1)
	
	for j in range(len(x)):
	    if j < NGas:
	        Typ[j] = 0 # Gas
	    if ((j >= NGas) & (j < N_particles)):
	        Typ[j] = 1 # DM
	
	x = np.round(x, 5)
	y = np.round(y, 5)
	z = np.round(z, 5)

	vx = np.round(vx, 5)
	vy = np.round(vy, 5)
	vz = np.round(vz, 5)
	
	print('N_tot = ', len(x))

	x = x.astype(np.float32)
	y = y.astype(np.float32)
	z = z.astype(np.float32)

	vx = vx.astype(np.float32)
	vy = vy.astype(np.float32)
	vz = vz.astype(np.float32)

	mass = mass.astype(np.float32)
	u = u.astype(np.float32)
	h = h.astype(np.float32)
	epsilon = epsilon.astype(np.float32)
	
	Typ = Typ.astype(np.int32)
	
	print('uuuu = ', u)
	print('uuuu.shape = ', u.shape)

	# Save the arrays to a binary file:
	N_tot = N_particles + N_blank
	num = '000' #str(int(np.floor(N_tot/1000)))
	with open('IC_' + num + 'k.bin', "wb") as file:
	    file.write(Typ.tobytes())
	    file.write(x.tobytes())
	    file.write(y.tobytes())
	    file.write(z.tobytes())
	    
	    file.write(vx.tobytes())
	    file.write(vy.tobytes())
	    file.write(vz.tobytes())
	    
	    file.write(mass.tobytes())
	    file.write(h.tobytes())
	    file.write(epsilon.tobytes())
	    print('u before writing:', u)
	    file.write(u.tobytes())

	print()
	print(f'Total elapse time in seconds = {time.time() - TT}')


