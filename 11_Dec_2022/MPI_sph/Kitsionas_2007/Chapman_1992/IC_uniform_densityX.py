
# Solution of the isothermal Lane-Emden equation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
np.random.random(42)
import pickle
from libsx import *
import time

TA = time.time()


res = []
count = 0

N = 10000

while count < N:

	x = 1. - 2. * np.random.random()
	y = 1. - 2. * np.random.random()
	z = 1. - 2. * np.random.random()

	if (x**2 + y**2 + z**2)**0.5 <= 1.:
		res.append([x, y, z])
		count += 1

print('Total number of particles inside the sphere = ', count)

res = np.array(res)

x = res[:, 0]
y = res[:, 1]
z = res[:, 2]

N = res.shape[0]


M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 225. * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
R_0 = 1.94 # pc
UnitRadius_in_cm = R_0 * 3.086e18    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6

print('unitTime_in_Myr = ', unitTime_in_Myr)
print('unitVelocity = ', unitVelocity)
print('UnitMass_in_g = ', UnitMass_in_g)
print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)

G = grav_const_in_cgs

#Mcld = 50. * M_sun

#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
T_0 = 54. # K, see Table_1 in Anathpindika - 2009 - II

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.7
mH2 = muu * mH

c_0 = (kB * T_0 / mH2)**0.5

print('Sound speed (cm/s) = ', c_0)
print('Sound speed (m/s) = ', c_0/100.)
print('Sound speed (km/s) = ', c_0/100000.)
print()
#--------------------------


#------- Prepare the IC to output -------

m = np.ones(N) / N
h = do_smoothingX((res, res)) # We don't save this one as this is the h for only one of the clouds.
rho = getDensity(res, m, h)

print('rho = ', np.sort(rho)*UnitDensity_in_cgs)
#print('mean(rho) = ', np.mean(rho)*UnitDensity_in_cgs)

hB = np.median(h) # the two clouds will be separated by 2*hB

res2 = res.copy()
res2[:, 0] += (2.*1.0 + 2.*hB) # 1.0 is the radius of the cloud !

res12 = np.vstack((res, res2))

Mach = 3.
vel_ref = Mach * c_0 # The speed of each cloud. Note that the clouds are in a collision course so v1 = -v2.

v_cld_1 = np.zeros_like(res)
v_cld_2 = v_cld_1.copy()

v_cld_1[:, 0] = vel_ref
v_cld_2[:, 0] = -vel_ref

vel = np.vstack((v_cld_1, v_cld_2))

xx, yy = res12[:, 0], res12[:, 1]

print()
print(res12.shape)

print('Elapsed time = ', time.time() - TA)

plt.figure(figsize = (14, 6))
plt.scatter(xx, yy, s = 1, color = 'k')
plt.show()

#---- Output to a file ------
h = np.hstack((h, h))
m = np.hstack((m, m))
print('h.shape = ', h.shape)
print('m.shape = ', m.shape)

dictx = {'r': res12, 'v': vel, 'h': h, 'm': m}

with open('Data_Uniform_Initial_rho.pkl', 'wb') as f:
	pickle.dump(dictx, f)
#----------------------------


#------- check density profile -------
r = (x*x + y*y + z*z)**0.5

plt.scatter(r, rho, s = 1, color = 'black')
plt.show()
#-------------------------------------





