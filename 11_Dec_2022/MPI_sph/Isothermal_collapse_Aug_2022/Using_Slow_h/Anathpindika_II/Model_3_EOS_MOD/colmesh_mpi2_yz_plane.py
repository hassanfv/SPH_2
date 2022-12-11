

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time
from mpi4py import MPI
from libsx import *


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
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 0.84 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6

print('unitTime_in_Myr = ', unitTime_in_Myr)

#filez = np.sort(glob.glob('./Outputs_5k/*.pkl'))
filez = np.sort(glob.glob('./Outputs_19k_RAND_T_ps_10/*.pkl'))

j = 670

#filez = np.sort(glob.glob('./Outputs/*.pkl'))

#with open('Uniform_density_sphere.pkl', 'rb') as f:
with open(filez[j], 'rb') as f:
	data = pickle.load(f)

r = pos = data['pos']

xx = pos[:, 0]
yy = pos[:, 1]
zz = pos[:, 2]

#if rank == 0:
	#plt.scatter(xx, yy, s = 0.1)
	#plt.show()

N = r.shape[0]
m = 1.0 / (N/2) + np.zeros(N) # Note that m should be calculated like this !
#m = np.hstack((m, m))

h = data['h'] #do_smoothingX((r, r))

rho = data['rho'] * UnitDensity_in_cgs #getDensity(r, m, h) * UnitDensity_in_cgs

print('rho = ', np.sort(rho))


#N = pos.shape[0]
#MSPH = 1.0
#m = np.zeros(N) + MSPH/N

#x = [-1.00, 3.2]
x = [-0.00, 2.0]
y = [-1.00, 1.30]
z = [-1.0, 1.0]

dx = dy = dz = 0.02

xarr = np.arange(x[0]-dx, x[1], dx)
yarr = np.arange(y[0]-dy, y[1], dy)
zarr = np.arange(z[0]-dz, z[1], dz)


print(len(xarr) * len(yarr) * len(zarr))


Nz = len(zarr)
#------- used in MPI --------
count = Nz // nCPUs
remainder = Nz % nCPUs

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

		for j in range(len(yarr)):
			
			s = 0.
			for k in range(len(xarr)):
				
				r = np.array([xarr[k], yarr[j], zarr[i]])
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






