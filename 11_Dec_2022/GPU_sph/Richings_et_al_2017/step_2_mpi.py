
import numpy as np
from mpi4py import MPI
import os
from libsx2_2t import *
import pickle
import time
import matplotlib.pyplot as plt

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

with open('Grid.pkl', 'rb') as f:
	data = pickle.load(f)

r_in_kpc = data['r']
LBox_in_kpc = data['L'] # half with of the box.
mSPH_in_g = data['mSPH'] # mass of single SPH particle.

N = r_in_kpc.shape[0]

MSun = 1.98892e33 # g
M_tot_in_g = N * mSPH_in_g
LBox_in_cm = LBox_in_kpc * 1000. * 3.086e18 # multiplied by 1000 because it is kpc.
r_in_cm = r_in_kpc * 1000. * 3.086e18

#--- Normalizing to unit code ----
unitMass_in_g = M_tot_in_g
unitLength_in_cm = LBox_in_cm

UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3
print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')
mH = 1.6726e-24 # gram
unit_nH_in_per_cubic_cm = UnitDensity_in_cgs / mH

Mcld = M_tot_in_g / unitMass_in_g
Lcld = LBox_in_cm / unitLength_in_cm
r = r_in_cm / unitLength_in_cm

m = np.ones(N) * 1.0/N


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
local_h = smoothing_length_mpi(nbeg, nend, r)
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


#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = getDensity_mpi(nbeg, nend, r, m, h)
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



if rank == 0:
	print()
	print('nH in per cm^3 = ', np.sort(rho)*unit_nH_in_per_cubic_cm)

	dictx = {'r': r, 'm': m, 'h': h, 'rho': rho}
	
	num = str(int(np.floor(len(m)/1000)))
	
	with open('Main_IC_Grid_' + str(num) + 'k.pkl', 'wb') as f:
		pickle.dump(dictx, f)
	
	#os.remove('tmp1.pkl')
	
	print(r.shape)
	print()
	print('****************************')
	print('Step_2 Successfully Done !!!')
	print('****************************')
	#----------------------------


