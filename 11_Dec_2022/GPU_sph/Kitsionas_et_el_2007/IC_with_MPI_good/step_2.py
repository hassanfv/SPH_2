
# Solution of the isothermal Lane-Emden equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import pickle
from libsx2_2t import *
import time
import h5py

np.random.random(42)

TA = time.time()

# See: https://scicomp.stackexchange.com/questions/2424/solving-non-linear-singular-ode-with-scipy-odeint-odepack

def dSdx(x, S):
	
	y1, y2 = S
	
	return [y2, -2./x * y2 + np.exp(-y1)]


y1_0 = 0.
y2_0 = 0.
S_0 = (y1_0, y2_0)


x = np.linspace(.00001, 10., 10000)

sol = odeint(dSdx, y0 = S_0, t = x, tfirst = True)

y1_sol = sol.T[0]
y2_sol = sol.T[1]


#----- mu_from_ksi
def mu_from_ksi(x, y2_sol, ksi): #y2_sol is d_psi/d_ksi

	# finding the closest value
	x1 = x - ksi
	nx = np.where(x1 > 0.)[0]
	
	return ksi * ksi * y2_sol[nx[0] - 1]


#----- ksi_from_mu
def ksi_from_mu(x, y2_sol, mu):

	mu1 = x * x * y2_sol - mu
	nx = np.where(mu1 > 0.)[0]
	
	return x[nx[0]-1]


#----- psi_from_ksi
def psi_from_ksi(x, y1_sol, ksi):

	x1 = x - ksi
	nx = np.where(x1 > 0.)[0]
	
	return y1_sol[nx[0] - 1]


M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
G = grav_const_in_cgs


Mcld = 10. * M_sun  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

thetax = 3.0 # We choose this value for the xsi.

#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
gamma = 5./3.   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
T_0 = 35. # K, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.35  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mH2 = muu * mH

c_0 = (kB * T_0 / mH2)**0.5

print('Sound speed (cm/s) = ', c_0)
print('Sound speed (m/s) = ', c_0/100.)
print('Sound speed (km/s) = ', c_0/100000.)
print('kB/mH2 = ', kB/mH2)
print()
#--------------------------

#------ Central density ------
rho_0 = c_0**6 / 4./np.pi/G**3/Mcld**2 * (mu_from_ksi(x, y2_sol, thetax))**2

print('Central density (g/cm^3) = ', rho_0)
print('Central density (cm^-3) = ', rho_0/mH)
#-----------------------------


#------- Cloud radius --------
Rcld = c_0/(4.*np.pi*G*rho_0)**0.5 * thetax

print('Cloud radius (cm) = ', Rcld)
print('Cloud radius (pc) = ', Rcld/3.086e18)
#-----------------------------



#============ The Lane-Emden eq. solution ==============
delta_ksi = 0.02
ksi_grid = np.arange(0.01, thetax, delta_ksi)

#--- rho for each ksi:
rho_r = np.zeros_like(ksi_grid)

for i in range(len(ksi_grid)):

	psi_t = psi_from_ksi(x, y1_sol, ksi_grid[i])
	rho_r[i] = rho_0 * np.exp(-psi_t)

Lscale = c_0 / (4. * np.pi * G * rho_0)**0.5  # see eq. A34 in Turner et al. 1995.
delta_r = Lscale * delta_ksi

#plt.scatter(ksi_grid, rho_r, s = 1)
#plt.show()

#=======================================================
#--------- WE DO EVERYTHING IN PHYSICAL UNITS ----------
#=======================================================

#------- Reading the uniform Sphere of points ----------
with open('Uniform_Sphere.pkl', 'rb') as f:
	r_uniform = pickle.load(f)

r_uniform = r_uniform * Rcld # in cm
#-------------------------------------------------------

N_uniform = Npart = r_uniform.shape[0]

#------ Calculating the mass of each particle ----------
m = Mcld / (4./3.) / np.pi / Rcld**3 / Npart
#-------------------------------------------------------

res = []

delta = (1./Npart)**(1./3.) * Rcld # Interparticle separation in uniform distribution of particles.

TT1 = time.time()

for i in range(N_uniform):

	# pick a particle from the uniform sphere and calculate its r, theta, and phi:
	xt = r_uniform[i, 0]
	yt = r_uniform[i, 1]
	zt = r_uniform[i, 2]

	r_unif_t = (xt*xt + yt*yt + zt*zt)**0.5
	
	# calculating the corresponding theta & phi
	theta = np.arccos(zt/r_unif_t) # the angle from z-axis
	phi = 2.0*np.pi*np.random.random() #np.arctan(yt/xt)
	
	Mass_in_r_unif_t = 4. * np.pi * r_unif_t**3 * N_uniform * m / 3.

	# now we find the best index for which the summation gets almost equal to Mass_in_r_unif_t
	k = 0
	s = 0.
	while s < Mass_in_r_unif_t:
		
		s += 4. * np.pi * (Lscale * ksi_grid[k])**2 * rho_r[k] * delta_r
		k += 1
	
	r_dist = ksi_grid[k-1] * Lscale
	
	x_dist = r_dist * np.cos(phi) * np.sin(theta)
	y_dist = r_dist * np.sin(phi) * np.sin(theta)
	z_dist = r_dist * np.cos(theta)
	
	#----- Adding some amount of disorder to the particle positions ----
	rnd = np.random.random()
	if rnd > 0.5:
		sign = 1.0
	else:
		sign = -1.0

	rnd = np.random.random()
	if rnd < 1./3.:
		x_dist += sign * delta
	if (rnd >= 1./3.) & (rnd <= 2./3.):
		y_dist += sign * delta
	if rnd > 2./3.:
		z_dist += sign * delta

	res.append([x_dist, y_dist, z_dist])

print('TT1 = ', time.time() - TT1)

res = np.array(res)

r_uniform = r_uniform / Rcld

res = res / Rcld

print(res.shape)

dictx = {'r': res, 'rho_cen': rho_0, 'c_0': c_0, 'gamma': gamma, 'Rcld_in_pc': Rcld/3.086e18, 
	 'Rcld_in_cm': Rcld, 'Mcld_in_g': Mcld, 'mu': muu, 'grav_const_in_cgs': grav_const_in_cgs}

with open('main_IC_tmp1.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#----------------------------




