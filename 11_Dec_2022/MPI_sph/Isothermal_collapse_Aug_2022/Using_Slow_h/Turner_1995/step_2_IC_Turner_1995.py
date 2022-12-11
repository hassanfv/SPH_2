
# Solution of the isothermal Lane-Emden equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import pickle
from libsx import *
import time

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


Mcld = 225. * M_sun  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

thetax = 3.0 # We choose this value for the xsi.

#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
T_0 = 100. # K, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.40  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mH2 = muu * mH

c_0 = (kB * T_0 / mH2)**0.5

print('Sound speed (cm/s) = ', c_0)
print('Sound speed (m/s) = ', c_0/100.)
print('Sound speed (km/s) = ', c_0/100000.)
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

for i in range(N_uniform):

	# pick a particle from the uniform sphere and calculate its r, theta, and phi:
	xt = r_uniform[i, 0]
	yt = r_uniform[i, 1]
	zt = r_uniform[i, 2]

	r_unif_t = (xt*xt + yt*yt + zt*zt)**0.5
	
	# calculating the corresponding theta & phi
	theta = np.arccos(zt/r_unif_t) # the angle from z-axis
	phi = np.arctan(yt/xt)
	
	if (xt < 0.) & (yt > 0.):
		phi += np.pi
	
	elif (xt < 0.) & (yt < 0.):
		phi += np.pi
	
	elif (xt > 0.) & (yt < 0.):
		phi += 2.*np.pi
		

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

	res.append([x_dist, y_dist, z_dist])


res = np.array(res)

r_uniform = r_uniform / Rcld
xu, yu, zu = r_uniform[:, 0], r_uniform[:, 1], r_uniform[:, 2]

res = res / Rcld

x, y, z = res[:, 0], res[:, 1], res[:, 2]

#plt.figure().add_subplot(111, projection='3d').scatter(x, y, z, s = 1.);
#plt.show()


#------- Prepare the IC to output -------

m = np.ones(N_uniform) / N_uniform
h = do_smoothingX((res, res)) # We don't save this one as this is the h for only one of the clouds.
rho = getDensity(res, m, h)

hB = 0.0 #np.median(h) # the two clouds will be separated by 2*hB

res2 = res.copy()
res2[:, 0] += (2.*1.0 + 2.*hB) # 1.0 is the radius of the cloud !



#--- Applying the impact parameter on one of the clouds ---
b_param = 0.1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res[:, 1] = -b_param + res[:, 1] # see Turner et al - 1995
res2[:, 1] = b_param + res2[:, 1] # see Turner et al - 1995
#----------------------------------------------------------

res12 = np.vstack((res, res2))

Mach = 3.0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vel_ref = 1.7e5 # !!!!!!!!!!!!!!!!!!!!!!!!!!!! Mach * c_0 # The speed of each cloud. Note that the clouds are in a collision course so v1 = -v2.

v_cld_1 = np.zeros_like(res)
v_cld_2 = v_cld_1.copy()

v_cld_1[:, 0] = vel_ref
v_cld_2[:, 0] = -vel_ref

vel = np.vstack((v_cld_1, v_cld_2))

xx, yy = res12[:, 0], res12[:, 1]

print()
print(res.shape)
print(res12.shape)

print('Elapsed time = ', time.time() - TA)

plt.figure(figsize = (14, 6))
plt.scatter(xx, yy, s = 1, color = 'k')
plt.show()

#---- Output to a file ------
num = str(int(np.floor(len(h)/1000))) # Used for out put file name.
h = np.hstack((h, h))
print('h.shape = ', h.shape)
m = np.hstack((m, m))

print(res12.shape)

dictx = {'r': res12, 'v': vel, 'h': h, 'm': m, 'rho_cen': rho_0,
	 'Rcld_in_pc': Rcld/3.086e18, 'Mcld': Mcld, 'mu': muu, 'Mach': Mach} # rho_cen = central density.


with open('IC_Kitsionas_' + num + 'k.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#----------------------------


#------- check density profile -------
r = (x*x + (y+b_param)**2 + z*z)**0.5

plt.scatter(r, rho, s = 1, color = 'black')
plt.show()
#-------------------------------------





