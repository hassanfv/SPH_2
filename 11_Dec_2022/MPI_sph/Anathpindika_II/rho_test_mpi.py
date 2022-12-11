
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import time
from libsx import *
from mpi4py import MPI


T111 = time.time()

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 0.8 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6

print('unitTime_in_Myr = ', unitTime_in_Myr)


mH = 1.6726e-24 # gram
mH2 = 2.7 * mH


filz = np.sort(glob.glob('./Outputs/*.pkl'))
filz = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/AWS_16_April_Done/Outputs/*.pkl'))

j = 0

with open(filz[j], 'rb') as f:
	data = pickle.load(f)


r = data['pos']
t = data['current_t']
m = data['m']

N = r.shape[0]

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
	Th1 = time.time()
#-------- h (initial) -------
local_h = smoothing_length_mpi(nbeg, nend, r)  # This plays the role of the initial h so that the code can start !
h = 0 # This is just a placeholder. Its absence would crash this line :h = comm.bcast(h, root = 0) for CPUs other than rank = 0.

if rank == 0:
	h = local_h
	for i in range(1, nCPUs):
		h_tmp = comm.recv(source = i)
		h = np.concatenate((h, h_tmp))
else:
	comm.send(local_h, dest = 0)

h = comm.bcast(h, root = 0)
#----------------------------
if rank == 0:
	print('Th1 = ', time.time() - Th1)




#-------- rho ---------
if rank == 0:
	Trho = time.time()

local_rho = getDensity_mpi(nbeg, nend, r, m, h)
rho = 0.0 # This is just a placeholder. Its absence would crash this line :rho = comm.bcast(rho, root = 0) for CPUs other than rank = 0.

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



nH = rho * UnitDensity_in_cgs / mH

if rank == 0:
	print()
	print('Current time in Myr = ', t * unitTime_in_Myr)
	print('rho (g/cm3) = ', np.sort(rho * UnitDensity_in_cgs))
	print('nH = ', np.sort(nH))
	print('m = ', np.sort(m))

print()
print('Total Elapsed time = ', time.time() - T111)







