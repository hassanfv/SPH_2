
import numpy as np
from coolHeat2libs import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

#--- Scaling From Marinho et al. (2000) -----
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # == 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
#--------------------------------------------

u_vect = np.logspace(np.log10(0.1), np.log10(1340.) ,  10) # in code unit
rho_vect = np.logspace(np.log10(1e-7), np.log10(112.), 10) # in code unit

N = len(u_vect)

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

TA = time.time()

dt = 0.005

#===== uAfterCoolingGrid_mpi
def uAfterCoolingGrid_mpi(nbeg, nend, rho_vect, u_vect, dt):

	N = len(u_vect)
	M = nend - nbeg
	res = np.zeros((M*N, 4))

	k = 0

	for i in range(nbeg, nend): # u loop

		for j in range(N): # rho loop
		
			ut_cgs = u_vect[i] * Unit_u_in_cgs
			rhot_cgs = rho_vect[j] * UnitDensity_in_cgs
			dt_cgs = dt * UnitTime_in_s

			ucool = DoCooling_M(rhot_cgs, ut_cgs, dt_cgs) / Unit_u_in_cgs
			
			res[k, 0] = rho_vect[j]
			res[k, 1] = u_vect[i]
			res[k, 2] = ucool
			res[k, 3] = dt
			
			k += 1

	return res



#-------- u ---------
local_u = uAfterCoolingGrid_mpi(nbeg, nend, rho_vect, u_vect, dt)

if rank == 0:
	u = local_u
	for i in range(1, nCPUs):
		utmp = comm.recv(source = i)
		u = np.vstack((u, utmp))
else:
	comm.send(local_u, dest = 0)
#----------------------

if rank == 0:
	with open('u_after_cool_MPI_dt_0.005.pkl', 'wb') as f:
		pickle.dump(u, f)

print('TA = ', time.time() - TA)






