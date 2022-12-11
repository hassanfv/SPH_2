
# This is in the x-z plane.

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time
from mpi4py import MPI


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



M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 400.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 2.0 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6



filez = glob.glob('./Outputs_/*.pkl')
filez = np.sort(glob.glob('/mnt/Linux_Shared_Folder_2022/Outputs_Model_3_Anathpindika_2009/*.pkl'))

j = 670

with open(filez[j], 'rb') as f:
	data = pickle.load(f)

pos = data['pos']

xx = pos[:, 0]
yy = pos[:, 1]
zz = pos[:, 2]
t = data['current_t']

N = pos.shape[0]
MSPH = 1.0
m = np.zeros(N) + MSPH/N


x = [-1.1, 1.1]
y = z = [-1.1, 1.1]

dx = dy = dz = 0.01

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




#------------ h -------------
with open('hh.pkl', 'rb') as f:
	h = pickle.load(f)
#----------------------------


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


if rank == 0:
	with open('Nxy.pkl', 'wb') as f:
		pickle.dump(rho, f)





