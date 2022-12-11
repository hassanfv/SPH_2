
# The difference with _v3 is that here the particles can have different mass and the gravitational acceleration calculation is accordingly modified.
# The difference with _v2 is that here we incorporate shear viscosity by using the Balsara switch.
# The difference with previous version is that here we separated u and u_previous, ut_previous updates separately. See below.
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx2_2t import *
from mpi4py import MPI
from shear_test3_t_del import *


mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
mH2 = 2.7 * mH

M_sun = 1.98992e+33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

G = 1.0

gamma = 5./3.

UnitRadius_in_pc = 2.0
UnitRadius_in_cm = 3.086e18 * UnitRadius_in_pc

UnitMass_in_g = 10.0 * M_sun
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs

#----- P_polytrop_mpi (Anathpindika - 2009 - II)
@njit
def P_polytrop_mpi(nbeg, nend, rho, T_cld, T_ps, T_0):

	M = nend - nbeg
	P_res = np.zeros(M)

	#mH = 1.6726e-24 # gram
	#kB = 1.3807e-16  # cm2 g s-2 K-1
	#mH2 = 2.7 * mH

	kBmH2 = kB/mH2
	
	for i in range(nbeg, nend):
		
		rhot = rho[i]*UnitDensity_in_cgs
		
		if rhot <= 1.0e-21:
			P_res[i-nbeg] = rhot * kBmH2 * T_cld

		if (rhot > 1.0e-21) & (rhot <= 2.0e-21):
			P_res[i-nbeg] = rhot * kBmH2 * gamma * T_cld * (rhot/2.0e-21)**(gamma - 1.0)

		if (rhot > 2.0e-21) & (rhot <= 1.0e-18):
			P_res[i-nbeg] = rhot * kBmH2 * T_ps
		
		if rhot > 1.0e-18:
			P_res[i-nbeg] = rhot * kBmH2 * T_0 * (1.0 + gamma * (rhot/1.0e-14)**(gamma - 1.0))
	
	P_res = P_res / Unit_P_in_cgs

	return P_res




np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()


with open('hydroData_130k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

v = data['v']
vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

rho = data['rho']
P = data['P']
c = data['c']
h = data['h']
m = data['m']
#divV = data['divV']
#curlV = data['curlV']
alpha = data['alpha']


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

T_cld = T_ps = T_0 = 10.0


if rank == 0:
	TT = time.time()
#--------- p ----------
P = 0.
local_P = P_polytrop_mpi(nbeg, nend, rho, T_cld, T_ps, T_0)

if rank == 0:
	P = local_P
	for i in range(1, nCPUs):
		Ptmp = comm.recv(source = i)
		P = np.concatenate((P, Ptmp))
else:
	comm.send(local_P, dest = 0)

P = comm.bcast(P, root = 0)
#----------------------

if rank == 0:
	print('TT = ', time.time() - TT)
	print()
	print(P)






