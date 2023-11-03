
import numpy as np
#import matplotlib.pyplot as plt
import time
from libsx import *
# import pandas as pd
import pickle
from mpi4py import MPI
import struct
import random

np.random.seed(42)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

if rank == 0:
    TT = time.time()

# Constants
Msun = 1.989e33  # Solar mass in grams
grav_const_in_cgs = G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
mH = 1.6726219e-24  # Proton mass in grams
clight = 29979245800.  # cm/s
cm_to_kpc = 3.086e21

mu = 0.61

sigma = 200. * 1000. * 100. # cm/s =====> 200 km/s - See eq.1 in Richings et al - 2018


L_box = 0.97

mSPH_high = 60 # Msun
N_high = 680000 # This is the number of particles for only the high res octant!

mSPH_low = 400.0 # Msun
N_low = 100000 # This is the number of particles for each low-res octant. In total, it will be 7 * N_low for all 7 low-res octants!


stp_high = L_box / 2.0 / int(N_high**(1./3.))

stp_low = L_box / 2.0 / int(N_low**(1./3.))

r = []
m_in_Msun = []

for x in np.arange(0, L_box/2, stp_high):
  for y in np.arange(0, L_box/2, stp_high):
    for z in np.arange(0, L_box/2, stp_high):
    
      xx = random.uniform(x, x + stp_high)
      yy = random.uniform(y, y + stp_high)
      zz = random.uniform(z, z + stp_high)
    
      r.append([xx, yy, zz])
      m_in_Msun.append(mSPH_high)


# Fill the rest of the box with low-res particles
for x in [0, L_box/2]:
  for y in [0, L_box/2]:
    for z in [0, L_box/2]:
      if x == y == z == 0:
        # Skip the high-res octant
        continue
      for xi in np.arange(x, x + L_box/2, stp_low):
        for yi in np.arange(y, y + L_box/2, stp_low):
          for zi in np.arange(z, z + L_box/2, stp_low):
          
            xx = random.uniform(xi, xi + stp_low)
            yy = random.uniform(yi, yi + stp_low)
            zz = random.uniform(zi, zi + stp_low)
          
            r.append((xx, yy, zz))
            m_in_Msun.append(mSPH_low)

r = np.array(r) # # The unit of length is 1 kpc!
mass_in_Msun = np.array(m_in_Msun) # mass of the particles in Msun. Will be used to define UnitMass!

M_tot_in_Msun = np.sum(mass_in_Msun)

if rank == 0:
  print(f'total mass of the system = {M_tot_in_Msun:.5E} Msun')

#===============================================================
#============ Setting the units of the simulation ==============
#===============================================================
unitMass_in_g = M_tot_in_Msun * Msun
unitLength_in_cm = cm_to_kpc  # This is 1 kpc in cm so our unit length is 1 kpc!
Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s

if rank == 0:
    print()
    print(f'Unit Length = {unitLength_in_cm} cm')
    print(f'Unit mass = {unitMass_in_g} grams')
    print(f'Unit_time_in_s = {unitTime_in_s:.3E} seconds')
    print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
    print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')
    print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')
    print(f'unitVelocity_in_km/s = {round(unitVelocity_in_cm_per_s/100000., 2)} km/s')
    print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.4E}')
    print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')
    print()
# ------------------------------------------------------

L_box_in_kpc = L_box

# Since unit_length is 1 kpc therefore this r is already in unit length!
r = r - L_box_in_kpc / 2.0 # Subtracted by L_box_in_kpc/2 so that it is symmetric around 0.0.

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

if rank == 0:
    print('r.shape = ', r.shape)

v = np.zeros_like(r)

vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

mass = mass_in_Msun / M_tot_in_Msun
if rank == 0:
    print('mass = ', mass)

N_tot = len(mass)

T_gas = 10000.
u = (3/2) * kB * T_gas / mu / mH
u = np.full(N_tot, u) / Unit_u_in_cgs

if rank == 0:
    print('u.shape = ', u.shape)
    print('u = ', u)



#plt.scatter(x, y, s = 0.01, color = 'k')
#plt.show()

#===== Used for Outflow injection =====
G = 1.0

Tou_in = 1.0
L_AGN = 1e46  # erg/s !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Note that we multiply u by mass because u is in per unit mass!!
L_AGN_code_unit = L_AGN / (Unit_u_in_cgs * unitMass_in_g / unitTime_in_s)
# So L_AGN is now in energy per unit mass per unit time.

clight_code_unit = clight / unitVelocity_in_cm_per_s

vin = 30000. * 1000. * 100.
vin_in_code_unit = vin / unitVelocity_in_cm_per_s

M_dot_in_code_unit = Tou_in * L_AGN_code_unit / clight_code_unit / vin_in_code_unit

u_for_10K_Temp = (3/2) * kB * T_gas / mu / mH / Unit_u_in_cgs
#=======================================

dt = 8e-7 #----> used to estimate the multiplier!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

multiplier = 1.0
m_sph_high_res = np.min(mass) * multiplier  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if rank == 0:
  print()
  print('M_dot_in_code_unit = ', M_dot_in_code_unit)
  print('M_dot * dt / np.min(mass) = ', M_dot_in_code_unit * dt / np.min(mass))
  print('M_dot * dt / np.max(mass) = ', M_dot_in_code_unit * dt / np.max(mass))
  print('M_dot * dt / m_sph_high_res = ', M_dot_in_code_unit * dt / m_sph_high_res)
  print()



N = N_tot  # All are gas particles, so it is fine to use MPI to get h!

# ------- used in MPI --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
    nbeg = rank * (count + 1)
    nend = nbeg + count + 1
else:
    nbeg = rank * count + remainder
    nend = nbeg + count
# ----------------------------

if rank == 0:
    Th2 = time.time()
# --------- h (main) ---------
local_h = smoothing_length_mpi(nbeg, nend, r)
h = 0.0

if rank == 0:
    h = local_h
    for i in range(1, nCPUs):
        htmp = comm.recv(source=i)
        h = np.concatenate((h, htmp))
else:
    comm.send(local_h, dest=0)

h = comm.bcast(h, root=0)
comm.Barrier()
if rank == 0:
    print('Th2 = ', time.time() - Th2)
# ----------------------------

if rank == 0:
    print('h = ', h)


if rank == 0:
    epsilon = h.copy()

    Typ = np.full(N, 0)  # It is 0 because all N particles are gas particle!

    # *************************************************************************
    # *********** Updating the arrays for AGN outflow injection ***************
    # *************************************************************************

    # We will extend the arrays by N_blank = 10000, which means maximally 10000 outflow particles can be
    # injected without any issue. If we think it may exceed this value, we have to adjust this
    # value!

    N_blank = 20000  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! May need to change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    blank = np.zeros(N_blank)
    # Type of blank is set to -1 so that they are distinct from gas (Typ = 0) and collisionless (Typ = 1) particles!
    blank_Typ = np.full(N_blank, -1)

    Typ = np.hstack((Typ, blank_Typ))

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

    # Seting up BH particle at the center ====> index N is set at BH. Note that the last gas particle is at N-1 !
    Typ[N] = 1  # Since BH is a collisionless particle, its Type is set to 1 !

    x[N] = 0.0
    y[N] = 0.0
    z[N] = 0.0

    vx[N] = 0.0
    vy[N] = 0.0
    vz[N] = 0.0

    M_BH_in_g = 1e8 * Msun
    mass[N] = M_BH_in_g / unitMass_in_g
    u[N] = 0.0
    h[N] = 0.0
    # equivalent to 1 pc in code unit. Recall that unitLength is 1 kpc!
    epsilon[N] = 0.001

    x = np.round(x, 5)
    y = np.round(y, 5)
    z = np.round(z, 5)

    vx = np.round(vx, 5)
    vy = np.round(vy, 5)
    vz = np.round(vz, 5)

    Typ = Typ.astype(np.int32)

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
    N_tot = N + N_blank
    num = str(int(np.floor(N_tot/1000)))
    filename = 'IC_R_' + num + 'k.bin'
    with open(filename, "wb") as file:
        # Save the rest of the arrays:
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
        file.write(u.tobytes())

    # Saving the parameters and constants!
   
    sigma_in_code_unit = sigma / unitVelocity_in_cm_per_s
    
    print()
    print('M_dot_in_code_unit = ', M_dot_in_code_unit)
    print()
    

    with open('params.txt', 'w') as f:
        # Write each variable on its own line
        f.write(f'{filename}\n')
        f.write(f'{N_tot}\n')
        f.write(f'{G}\n')
        # Note that this will be multiplied by dt in the code. So dont get surprised by its high value!
        f.write(f'{L_AGN_code_unit}\n')
        # Note that this will be multiplied by dt in the code. So dont get surprised by its high value!
        f.write(f'{M_dot_in_code_unit}\n')
        f.write(f'{vin_in_code_unit}\n')
        f.write(f'{u_for_10K_Temp}\n')
        f.write(f'{m_sph_high_res}\n')
        f.write(f'{sigma_in_code_unit}\n')
        f.write(f'{UnitDensity_in_cgs}\n')
        f.write(f'{Unit_u_in_cgs}\n')
        f.write(f'{unitTime_in_s}\n')
        f.write(f'{unitLength_in_cm}\n')

    print(f'Total gas particle is {N} with N_blank = {N_blank}. So N_tot = {N_tot}')

    print()
    print(f'Total elapse time in seconds = {time.time() - TT}')



