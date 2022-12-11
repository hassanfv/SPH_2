
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from numba import njit
import time


#===== smooth_hX (non-parallel)
@njit(parallel=True)
def do_smoothingX(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j, 0] - subpos[i, 0]
            dy = pos[j, 1] - subpos[i, 1]
            dz = pos[j, 2] - subpos[i, 2]
            dist[j] = np.sqrt(dx**2 + dy**2 + dz**2)

        hres.append(np.sort(dist)[32])

    return np.array(hres) * 0.5




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
UnitMass_in_g = 400.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 2.0 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6



filez = np.sort(glob.glob('./Outputs/*.pkl'))

j = -1

with open(filez[j], 'rb') as f:
	data = pickle.load(f)

pos = data['pos']

xx = pos[:, 0]
yy = pos[:, 1]
zz = pos[:, 2]
t = data['current_t']

#----------------------------------------

#----------------------------------------

N = pos.shape[0]
MSPH = 1.0
m = np.zeros(N) + MSPH/N


x = [-2.25, 2.25]
y = z = [-1.5, 1.5]

dx = dy = dz = 0.1

xarr = np.arange(x[0]-dx, x[1], dx)
yarr = zarr = np.arange(y[0]-dy, y[1], dy)

print(len(xarr) * len(yarr) * len(zarr))


Th = time.time()
h = do_smoothingX((pos, pos))
print('Th = ', time.time() - Th)


#----- densityx
def densityx(m, WI):
	
	s = 0.
	N = len(m)
	for j in range(N):
	
		s += m[j] * WI[j]
	
	return s



Ts = time.time()

rho = np.zeros((len(xarr), len(yarr)))

for i in range(len(xarr)):

	for j in range(len(yarr)):
		
		s = 0.
		for k in range(len(zarr)):
			
			r = np.array([xarr[i], yarr[j], zarr[k]])
			hs = do_smoothingX_single(r, pos)
			
			WI = W_I(r, pos, hs, h)
			
			s += densityx(m, WI)
		
		rho[i, j] = s


print('Ts = ', time.time() - Ts)



with open('Nxy.pkl', 'wb') as f:
	pickle.dump(rho, f)



plt.figure(figsize = (12, 8))
plt.scatter(xx, yy, s = 0.1, color = 'black')

plt.xlim(-2.25, 2.25)
plt.ylim(-1.5, 1.5)

plt.show()









