
import time
import numpy as np
import random
import pickle

np.random.seed = 42

Npart = 1400000 # Play with this to get the desired number of particles!

grav_const_in_cgs = 6.6738e-8
MSun = 1.98892e33

Mcld_in_g = 1.0 * MSun                                     # The mass of the cloud
Rcld_in_cm = 4.99e16                                     # The initial radius of the cloud in cm
omega = 7.2e-13                                 # The initial angular velocity of the cloud in radians s^-1 
rho0 = 3.82e-18                                 # The initial average density
cs = 1.66e4  # this corrsponds to mu = 2.28 in (kB * T_0 / mH2)**0.5 (mH2 = muu * mH)  # The sound speed

# Calculating derived quantities
tff = np.sqrt(3*np.pi/(32*grav_const_in_cgs*rho0))                   # The free-fall time = 3.4e4 yr
tff_in_kyr = tff/3600/24/365.24/1000
print(f'tff = {tff:.2E} seconds')
print(f'tff = {tff_in_kyr:.2f} kyrs')

# Setting the units of the simulation
unitMass_in_g = Mcld_in_g
unitLength_in_cm = Rcld_in_cm
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s

print(f'Unit_time_in_s = {round(unitTime_in_s, 2)} seconds')
print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')

print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')

# calling things to code units
Rcld = Rcld_in_cm / unitLength_in_cm
Mcld = Mcld_in_g / unitMass_in_g
omega *= unitTime_in_s

UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3
print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')

# Arreaga-García et al (2007)

L = 2.1 # Length of the cube. It is fine to get it a bit larger as we will constrain by r<=1.
V = L**3
delta = (V/Npart)**(1./3.)

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

## Calculating particle velocities in rectangular coordinates

rxy = (pos[:,0]**2 + pos[:,1]**2)**0.5 
vel = np.zeros_like(pos)

vel[:,0] = -omega*pos[:,1]
vel[:,1] = omega*pos[:,0]

vel[:,2] = 0

wh = np.argwhere(np.isnan(vel)) # NaNs are handled here !
if len(wh) > 0:
    vel[wh] = 0.0


## Calculating particle masses

mp = Mcld / pos.shape[0]

# Imposing an m=2 density perturbation with an amplitude of 10 percent.
masses = mp * (1 + .1*((pos[:, 0]/rxy)**2 - (pos[:, 1]/rxy)**2))

wh = np.argwhere(np.isnan(masses)) # Fixes an issue with the particle at the origin
if len(wh) > 0:
    masses[wh] = mp

###################

#vel /= unitVelocity_in_cm_per_s

G = 1.0
NSample = pos.shape[0]
Rcld_in_pc = Rcld_in_cm/3.086e18
muu = 3.0
gamma = 5./3.
c_0 = 1.66e4 # cm/s # this corresponds to c_iso in Arrega et al - 2007
Mach = 0.0 # not applicable in this test.

#---- physical parameters used for GPU ----
#            0      1     2         3          4           5       6    7            8          9
paramz = [NSample, c_0, gamma, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach, grav_const_in_cgs, G]

u = np.zeros(len(masses)) + c_0**2 / unitVelocity_in_cm_per_s**2

dictx = {'r': pos, 'v': vel, 'm': masses, 'u': u, 'paramz': paramz}

with open('tmp1.pkl', 'wb') as f:
	pickle.dump(dictx, f)

print()
print(f'Total number of particles = {pos.shape[0]}')
print()
print('********************************')
print('Step_1 Successfully Finished !!!')
print('********************************')





