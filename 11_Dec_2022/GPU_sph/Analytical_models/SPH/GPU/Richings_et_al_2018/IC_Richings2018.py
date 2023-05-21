
import numpy as np
#import matplotlib.pyplot as plt
import time
from libsx import *
#import pandas as pd
import pickle
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

if rank == 0:
	TT = time.time()

# Constants
Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
mH = 1.6726219e-24  # Proton mass in grams
clight = 29979245800. # cm/s
cm_to_kpc = 3.086e21

N_particles = 10000 # desired number of SPH particles in the simulation.

nH = 10.0 # cm^-3
T_gas = 1e4 # K

mu = 0.61 # fully ionized gas with solar metallicity!
rho = mu * mH * nH

mSPH = 80.0 # M_sun
mSPH_in_g = 80.0 * Msun # grams
M_tot = N_particles * mSPH
M_tot_in_g = M_tot * Msun # Total mass in grams.

L_box_in_cm = (M_tot_in_g / rho)**(1./3.) # box size in cm.

L_box_in_kpc = L_box_in_cm / cm_to_kpc

if rank == 0:
	print(f'With nH = {nH}, and N_particles = {N_particles}, we get L_box = {round(L_box_in_kpc, 2)} kpc.')

# ------ Setting the units of the simulation ---------
grav_const_in_cgs = 6.67430e-8
unitMass_in_g = M_tot_in_g
unitLength_in_cm = cm_to_kpc # This is 1 kpc in cm so our unit length is 1 kpc!
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s
Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3

if rank == 0:
	print()
	print(f'Unit_time_in_s = {unitTime_in_s:.3E} seconds')
	print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
	print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')
	print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')
	print(f'unitVelocity_in_km/s = {round(unitVelocity_in_cm_per_s/100000., 2)} km/s')
	print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.4E}')
	print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')
	print()
#------------------------------------------------------

# Create a 3D grid of particles
r = np.random.uniform(0, L_box_in_kpc, (N_particles, 3)) # The unit of length is 1 kpc!

r = r - L_box_in_kpc / 2.0 # Since unit_length is 1 kpc therefore this r is already in unit length!

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]


if rank == 0:
	print(y[:100])


if rank == 0:
	print('r.shape = ', r.shape)

v = np.zeros_like(r)

vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

mass = np.full(N_particles, mSPH_in_g / M_tot_in_g)
if rank == 0:
	print('mass = ', mass)

u = (3/2) * kB * T_gas / mu / mH
u = np.full(N_particles, u) / Unit_u_in_cgs
if rank == 0:
	print('u.shape = ', u.shape)
	print('u = ', u)

#plt.scatter(r[:, 0], r[:, 1], s = 0.01, color = 'k')
#plt.show()

N = N_particles

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

if rank == 0:
	print('h = ', h)

if rank == 0:
	epsilon = h.copy()

	#*************************************************************************
	#*********** Updating the arrays for AGN outflow injection ***************
	#*************************************************************************

	# We will extend the arrays by N_blank = 10000, which means maximally 10000 outflow particles can be 
	# injected without any issue. If we think it may exceed this value, we have to adjust this
	# value!

	N_blank = 10000
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
	
	# Seting up BH particle at the center ====> The first index in all arrays represent the BH particle.
	x[0] = 0.0
	y[0] = 0.0
	z[0] = 0.0
	
	vx[0] = 0.0
	vy[0] = 0.0
	vz[0] = 0.0
	
	M_BH_in_g = 1e8 * Msun
	mass[0] = M_BH_in_g / unitMass_in_g
	u[0] = 0.0
	h[0] = 0.0
	epsilon[0] = 0.001 # equivalent to 1 pc in code unit. Recall that unitLength is 1 kpc!
	
	
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

	mass = mass.astype(np.float32)
	u = u.astype(np.float32)
	h = h.astype(np.float32)
	epsilon = epsilon.astype(np.float32)
	

	# Save the arrays to a binary file:
	N_tot = N_particles + N_blank
	num = str(int(np.floor(N_tot/1000)))
	with open('IC_Richings2018_' + num + 'k.bin', "wb") as file:
	    file.write(x.tobytes())
	    file.write(y.tobytes())
	    file.write(z.tobytes())
	    
	    file.write(vx.tobytes())
	    file.write(vy.tobytes())
	    file.write(vz.tobytes())
	    
	    file.write(mass.tobytes())
	    file.write(h.tobytes())
	    file.write(epsilon.tobytes())
	    file.write(u.tobytes())

	# Saving the parameters and constants!
	G = 1.0
	
	Tou_in = 1.0
	L_AGN = 1e45 # erg/s
	L_AGN_code_unit = L_AGN / (Unit_u_in_cgs * unitMass_in_g / unitTime_in_s) # Note that we multiply u by mass because u is in per unit mass!!
								                   # So L_AGN is now in energy per unit mass per unit time.
	clight_code_unit = clight / unitVelocity_in_cm_per_s
	vin = 30000. * 1000. * 100.
	vin_in_code_unit = vin / unitVelocity_in_cm_per_s

	M_dot_in_code_unit = Tou_in * L_AGN_code_unit / clight_code_unit / vin_in_code_unit
	u_for_10K_Temp = (3/2) * kB * T_gas / mu / mH / Unit_u_in_cgs
	m_sph_high_res = mSPH_in_g / unitMass_in_g

	with open('params.txt', 'w') as f:
		# Write each variable on its own line
		f.write(f'{N_tot}\n')
		f.write(f'{N_particles}\n')
		f.write(f'{N_blank}\n')
		f.write(f'{G}\n')
		f.write(f'{L_AGN_code_unit}\n')    # Note that this will be multiplied by dt in the code. So dont get surprised by its high value!
		f.write(f'{M_dot_in_code_unit}\n') # Note that this will be multiplied by dt in the code. So dont get surprised by its high value!
		f.write(f'{vin_in_code_unit}\n')
		f.write(f'{u_for_10K_Temp}\n')
		f.write(f'{m_sph_high_res}\n')
	

	print(f'Total gas particle is {N_particles} with N_blank = {N_blank}. So N_tot = {N_tot}')

	print()
	print(f'Total elapse time in seconds = {time.time() - TT}')


