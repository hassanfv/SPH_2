
# This is in the x-z plane.

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time


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



with open('Data_Turner_10000.pkl', 'rb') as f:
	data = pickle.load(f)

pos = data['r']

xx = pos[:, 0]
yy = pos[:, 1]
zz = pos[:, 2]
h = data['h']

N = pos.shape[0]
MSPH = 1.0
m = np.zeros(N) + MSPH/N


r = np.array([0., 0., 0.])
hs = do_smoothingX_single(r, pos)

WI = W_I(r, pos, hs, h)

rho_c = densityx(m, WI) * UnitDensity_in_cgs

print()
print('Central density (g/cm^3) = ', rho_c)


#---------- Barotropic (Kitsionas et al 2007) ----------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.7
mH2 = muu * mH

rho_grid = np.logspace(-1, 5, 1000) * rho_c

Mcld = 50. * M_sun
Tcld = 54. # K
c_0 = 20000. * (Mcld/M_sun)**0.25

rho_1 = 1e-14

P = np.zeros_like(rho_grid)
T = np.zeros_like(rho_grid)

for i in range(len(rho_grid)):

	rhot = rho_grid[i] # rho_grid is already in physical unit.
	c_s = (kB/mH2*Tcld)**0.5

	if rhot <= rho_c:

		P[i] = rhot * c_0**2
	
	if rhot > rho_c:
		P[i] = rhot * ((c_0**2 - c_s**2) * (rhot/rho_c)**(-2./3.) + c_s**2) * (1. + (rhot/rho_1)**(4./3.))**0.5
	
	#T[i] = mH2/kB*P[i]/rhot


#print(rho_grid)
#print('T = ', T)

gamma = 5./3.

#-------- Barotropic (Anathpindika - 2009) ---------
PP = np.zeros_like(rho_grid)
#TT = np.zeros_like(rho_grid)

for i in range(len(rho_grid)):

	rhot = rho_grid[i] # rho_grid is already in physical unit.

	if rhot <= rho_c:

		PP[i] = rhot * kB/mH2 * Tcld
	
	if rhot > rho_c:
		PP[i] = rhot * kB/mH2 * gamma * Tcld * (rhot/5.e-20)**(gamma-1.)
#---------------------------------------------------

plt.scatter(rho_grid, P, s = 0.5, color = 'k')
plt.scatter(rho_grid, PP, s = 0.5, color = 'blue')
plt.xscale('log')
plt.yscale('log')

plt.show()

#-------------------------------------------------------








		



