
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
np.random.seed(42)
random.seed(40)


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




file1 = open('BB.txt', 'r') # Taken from https://github.com/dhubber/seren/tree/master/src/ic
lines = file1.readlines()
file1.close()

res = []

for x in lines:

	x = x.split(' ')
	x[-1] = x[-1].split('\n')[0]

	resT = []
	for xt in x:
		if len(xt) > 1:
			resT.append(xt)

	x = resT
	res.append([float(xt) for xt in x])

res = np.array(res)

npart = 3000

resT = []
for j in range(npart):
	rand_index = random.randint(0, len(res)-1)
	resT.append(list(res[rand_index, :]))

rT = np.array(resT) * 100.

print(res.shape)


#----------- Uniform circular velocity around z-axis -----------

resx = rT[:, 0]
resy = rT[:, 1]
resz = rT[:, 2]

N = len(resx)

rr = np.sqrt(resx**2 + resy**2 + resz**2) # (converting to physical unit because omega is in physical unit, i.e. rad/s)
#omega = 7.2e-13 # rad/s (angular velocity; See Gingold & Monaghan 1981).
omega = 1.6e-12 # rad/s (angular velocity; See Boss & Bodenheimer 1979).
vel = (rr * R) * omega # Here the velocities are in physical unit, i.e. in cm/s.
vel = vel/unitVelocity # The velocities are now in code unit.

vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)

for i in range(N):
	
	vx[i] = -omega * R * resy[i] / unitVelocity
	vy[i] =  omega * R * resx[i] / unitVelocity
	vz[i] = 0.

vx = vx.reshape((1, N))
vy = vy.reshape((1, N))
vz = vz.reshape((1, N))

v = np.hstack((vx.T, vy.T, vz.T))
#---------------------------------------------------------------


dictxT = {'r': rT, 'v': v}
with open('Boss_IC_' + str(int(npart)) + '.pkl', 'wb') as f:
	pickle.dump(dictxT, f)


#dictx = {'x':res[:, 0], 'y':res[:, 1], 'z':res[:, 2]}
#with open('Boss_IC_25600.pkl', 'wb') as f:
#	pickle.dump(dictx, f)


plt.figure(figsize = (9, 8))

plt.scatter(rT[:, 0], rT[:, 1], s = 0.1, color = 'black')

plt.show()
	



