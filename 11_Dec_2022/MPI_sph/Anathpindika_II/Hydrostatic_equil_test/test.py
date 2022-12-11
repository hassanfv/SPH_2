
M_sun = 1.989e33 # gram
grav_const_in_cgs = 6.67259e-8 #  cm3 g-1 s-2
UnitMass_in_g = 50.0 * M_sun       # !!!!!!!!!!!!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!
rB = 0.8 # pc
ksi = 3.
R_0 = rB/ksi
UnitRadius_in_cm = R_0 * 3.086e18 # cm (2.0 pc)    #!!!!!!!!!!!!!! CHANGE !!!!!!!!!!!!!!!!!!
UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm**3
Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm
Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs
unitVelocity = (grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm)**0.5

unitTime = (UnitRadius_in_cm**3/grav_const_in_cgs/UnitMass_in_g)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1.e6

print('unitTime_in_Myr = ', unitTime_in_Myr)
print('UnitDensity_in_cgs = ', UnitDensity_in_cgs)
