
import numpy as np


#===== pc_to_cm
def pc_to_cm(pc):

	return pc * 3.086e+18

#===== cm_to_pc
def cm_to_pc(cm):

	return cm / 3.086e+18




#--- Scaling From Marinho et al. (2000)---
M_sun = 1.989e33 # gram
UnitRadius_in_cm = 3.086e18 # i.e. 1 pc
UnitTime_in_s = 3.16e13 # == 1Myr
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2

UnitMass_in_g = UnitRadius_in_cm**3 / grav_const_in_cgs / UnitTime_in_s**2
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3

print('M = ', UnitMass_in_g)
print('M/M_sun = ', UnitMass_in_g/M_sun)
print('U = ', Unit_u_in_cgs)
print('RHO = ', UnitDensity_in_cgs)

