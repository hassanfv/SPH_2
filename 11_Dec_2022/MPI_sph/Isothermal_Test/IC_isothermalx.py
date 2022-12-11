
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(40)

M_sun = 1.989e33 # gram
G_cgs = 6.67259e-8 #  cm3 g-1 s-2

R = 3.2e16 # cm
M = 1. * M_sun

unitVelocity = (G_cgs * M / R)**0.5
#unitDensity = 3.*M/4./np.pi/R**3
unitDensity = M/R**3
unit_u = G_cgs*M/R
unitTime = (R**3/G_cgs/M)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6

print('unitVelocity = ', unitVelocity)
print('unitDensity = ', unitDensity)
print('unit_u = ', unit_u)
print('unitTime = ', unitTime)
print('unitTime_in_yr = ', unitTime_in_yr)
print('unitTime_in_Myr = ', unitTime_in_Myr)

nPart = 8000

res = []

M = 1.
rho0 = 3./4./np.pi

print('rho0_in_cgs = ', unitDensity * rho0)

for i in range(nPart):

	ksi1, ksi2, ksi3 = np.random.random(3)

	theta = np.arccos(1. - 2. * ksi2)
	phi = 2. * np.pi * ksi3
	
	r = (3.*M*ksi1/4./np.pi/rho0)**(1./3.)
	
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	
	res.append([x, y, z])


res = np.array(res)
resx = res[:, 0]
resy = res[:, 1]
resz = res[:, 2]

N = nPart

#----------- Uniform circular velocity around z-axis -----------
rr = np.sqrt(resx**2 + resy**2 + resz**2) # (converting to physical unit because omega is in physical unit, i.e. rad/s)
#omega = 7.2e-13 # rad/s (angular velocity; See Gingold & Monaghan 1981).
omega = 1.6e-12 #1.6e-12 # rad/s (angular velocity; See Boss & Bodenheimer 1979).
vel = (rr * R) * omega # Here the velocities are in physical unit, i.e. in cm/s.
vel = vel/unitVelocity # The velocities are now in code unit.

sin_T = resy.T / rr
cos_T = resx.T / rr

vx = np.abs(vel * sin_T)
vy = np.abs(vel * cos_T)
vz = 0.0 * vx

nregA = (resx.T >= 0.0) & (resy.T >= 0.0)

vx[nregA] = -vx[nregA]

nregB = (resx.T < 0.0) & (resy.T >= 0.0)
vx[nregB] = -vx[nregB]
vy[nregB] = -vy[nregB]

nregC = (resx.T < 0.0) & (resy.T < 0.0)
vy[nregC] = -vy[nregC]

vx = vx.reshape((1, N))
vy = vy.reshape((1, N))
vz = vz.reshape((1, N))

v = np.hstack((vx.T, vy.T, vz.T))
#---------------------------------------------------------------


#-------- Adding Azimuthally varying perturbation --------------
#------------ (see Sigalotti & Klapp 1997) ---------------------
m = np.zeros(N)

for i in range(N):

	phi = np.arctan(resy[i]/resx[i])

	m[i] = M/N * (1. + 0.5 * np.cos(2.*phi))
#---------------------------------------------------------------


print(np.sort(m))

plt.hist(m, bins = 15)
plt.show()

print('sum(m) = ', np.sum(m))

dictx = {'r': res, 'v': v, 'm': m}
with open('IsoData.pkl', 'wb') as f:
	pickle.dump(dictx, f)

plt.figure(figsize = (8, 8))

plt.scatter(res[:, 0], res[:, 1], s = 1, color = 'black')

#plt.xlim(-24, 24)
#plt.ylim(-24, 24)

plt.show()	







