
import time
import numpy as np
import random
import pickle
#import matplotlib.pyplot as plt
from photolibs import *

np.random.seed = 42

N_start = 180000 # Play with this to get the desired number of particles!

grav_const_in_cgs = 6.6738e-8
MSun = 1.98892e33
mH = 1.6726e-24
kB = 1.3807e-16
XH = 0.76

Mcld_in_g = 367.791 * MSun # !!!!!!!!!!!!!!!!!! calculated using the nH_M_R_estimator.py code here !
Rcld_in_pc = 30.0         # !!!!!!!!!!!!!!!!!!
Rcld_in_cm = Rcld_in_pc * 3.086e18

rho_0 = Mcld_in_g / (4./3.) / np.pi / Rcld_in_cm**3

nH_0 = XH * rho_0 / mH # cm^-1

print(f'rho_0 = {rho_0:.3E}')
print(f'nH_0 = {nH_0:.3f}')

T_0 = 10000. # K

muu = 1.22 # gas with atomic hydrogen + helium
c_s = (kB * T_0 / muu/mH)**0.5

print(f'c_s = ', c_s)

# Calculating derived quantities
tff = np.sqrt(3*np.pi/(32*grav_const_in_cgs*rho_0))
tff_in_kyr = tff/3600/24/365.24/1000
tff_in_Myr = tff/3600/24/365.24/1e6
print(f'tff = {tff:.2E} seconds')
print(f'tff = {tff_in_kyr:.2f} kyrs')
print(f'tff = {tff_in_Myr:.2f} Myrs')

# Setting the units of the simulation
unitMass_in_g = Mcld_in_g
unitLength_in_cm = Rcld_in_cm
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s

print(f'Unit_time_in_s = {round(unitTime_in_s, 2)} seconds')
print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')

print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')

Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.4E}')

# calling things to code units
Rcld = Rcld_in_cm / unitLength_in_cm
Mcld = Mcld_in_g / unitMass_in_g
#omega *= unitTime_in_s

UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3
print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')

# Arreaga-García et al (2007)
L = 2.1 # Length of the cube. It is fine to get it a bit larger as we will constrain by r<=1.
V = L**3
delta = (V/N_start)**(1./3.)

M = int(np.floor(L / delta))

pos = []

for i in range(-M, M):
    for j in range(-M, M):
        for k in range(-M, M):
            
            xt, yt, zt = 0.0+i*delta, 0.0+j*delta, 0.0+k*delta
            
            rnd = np.random.random()
            if rnd > 0.5:
                sign = 1.0
            else:
                sign = -1.0
            
            # Adding some amount of disorder
            rnd = np.random.random()
            if rnd < 1./3.:
                xt += sign * delta/4.
            if (rnd >= 1./3.) & (rnd <= 2./3.):
                yt += sign * delta/4.
            if rnd > 2./3.:
                zt += sign * delta/4.
            
            r = (xt*xt + yt*yt + zt*zt)**0.5
            
            if r <= 1.0:
                pos.append([xt, yt, zt])

pos = np.array(pos)
print(pos.shape)

Npart = pos.shape[0]
print(f'Number of particles in a single cloud = ', Npart)

## Calculating particle masses
masses = Mcld / Npart + np.zeros(Npart)

G = 1.0
gamma = 5./3.
Mach = 0.0 # Place holder !!! Please update it in step_3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MachVel = Mach*c_s / 100. / 1000. # km/s
#print(f'Mach = {Mach} corresponds to Velocity = {MachVel:.3f} km/s')
print()
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('PLEASE UPDATE MACH NUMBER IN step3.py !!!!!!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print()

#---- physical parameters used for GPU ----
#           0     1     2         3          4           5       6    7            8          9
paramz = [Npart, c_s, gamma, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach, grav_const_in_cgs, G]

Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.3f}')

u = 4000.0 + np.zeros(Npart) # in code unit # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! I used check4.py code here to find this value!
u_in_cgs = u * Unit_u_in_cgs # in cgs

T = convert_u_to_temp_h(u_in_cgs[0], nH_0, XH)

print(f'T = {T:.2f} K which corresponds to u (code unit) = {u[0]} and to u (cgs) = {u_in_cgs[0]:.4E}')

dictx = {'r': pos, 'm': masses, 'u': u, 'paramz': paramz}

with open('tmp1.pkl', 'wb') as f:
	pickle.dump(dictx, f)

print(f'Each SPH particle has a mass of {masses[0]*unitMass_in_g/MSun:.2E} M_sun')
print()
print(f'Total number of particles = {pos.shape[0]}')
print()
print('********************************')
print('Step_1 Successfully Finished !!!')
print('********************************')





