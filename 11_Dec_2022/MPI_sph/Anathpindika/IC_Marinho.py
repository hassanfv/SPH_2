
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(42)


M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 400.0 * M_sun        #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!
UnitRadius_in_cm = 2.0 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

print('unitVelocity = ', unitVelocity)

nPart = 60000

Rc = 1. # in code unit
# Note that the following calculation for r in "r = Rc * ksi1**0.5" line is for the asumption that M = 10 in code unit. See Marinho et al (2000; Appendix D.).

res = []

for i in range(nPart):

	ksi1, ksi2, ksi3 = np.random.random(3)

	r = Rc * ksi1**0.5
	theta = np.arccos(1. - 2. * ksi2)
	phi = 2. * np.pi * ksi3
	
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	
	res.append([x, y, z])


res1 = np.array(res)
res2 = res1.copy()
res1[:, 0] = res1[:, 0] + Rc # shifting the cloud 10. unit to the right side (i.e. positive x).
res2[:, 0] = res2[:, 0] - Rc # shifting the cloud 10. unit to the left side (i.e. negative x).

res = np.vstack((res1, res2))

vel1 = np.zeros_like(res1)
vel2 = np.zeros_like(res2)

# for T = 54. K the sound speed for isothermal gas of molecular hydrogen is 472.10 m/s or 47210. cm/s.
c_s = 83765.09 # cm/s  #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!
Mach = 25.             #!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!!!

vel1[:, 0] = -Mach * c_s #/ unitVelocity # We do this in the main code.
vel2[:, 0] =  Mach * c_s #/ unitVelocity

vel = np.vstack((vel1, vel2))


dictx = {'r': res, 'v': vel}

with open('Marinho_IC_' + str(int(nPart)) + '.pkl', 'wb') as f:
	pickle.dump(dictx, f)


plt.figure(figsize = (8, 8))

plt.scatter(res[:, 0], res[:, 1], s = 0.05, color = 'black')

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.show()	







