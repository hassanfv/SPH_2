
# plot in x-y plane

import numpy as np
import matplotlib.pyplot as plt
import pickle


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


with open('Nxy.pkl', 'rb') as f:
#with open('/mnt/Linux_Shared_Folder_2022/AWS_16_April_Done/Nxy.pkl', 'rb') as f:
	data = pickle.load(f)

rho = data['rho']
dx = data['dx']
mH = 1.6726e-24 # gram
muu = 2.7
mH2 = muu * mH

colden = rho * UnitDensity_in_cgs * dx * UnitRadius_in_cm # column density in g/cm2
colden2= colden / mH # column density in cm^-2

one_pc = 3.086e18 # cm
colden_Msun_pc2 = np.log10(colden * one_pc*one_pc / M_sun)

nx = np.where(colden_Msun_pc2 < 0.)
colden_Msun_pc2[nx] = 0.


print('max(colden in cm^-2) = ', np.max(colden2.flatten()))
print('max(colden in M_sun/pc^2) = ', np.max(colden_Msun_pc2.flatten()))


plt.figure(figsize = (12, 8))
plt.imshow(colden_Msun_pc2.T, cmap = 'rainbow_r')

plt.clim(0, 1.66)

plt.colorbar()

plt.savefig('fig_1.png')

plt.show()







