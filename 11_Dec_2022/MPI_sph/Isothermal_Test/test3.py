

M_sun = 1.989e33 # gram
G_cgs = 6.67259e-8 #  cm3 g-1 s-2

R = 3.2e16 # cm
M = 1. * M_sun

#---- For Unit ref. see Thacker et al - 2000 ------
unitVelocity = (G_cgs * M / R)**0.5
#unitDensity = 3.*M/4./np.pi/R**3
unitDensity = M/R**3
unit_u = G_cgs*M/R
unitTime = (R**3/G_cgs/M)**0.5
unitTime_in_yr = unitTime / 3600. / 24. / 365.25
unitTime_in_Myr = unitTime / 3600. / 24. / 365.25 / 1e6
unit_P = unitDensity * unit_u

print('unitVelocity = ', unitVelocity)
print('unitDensity = ', unitDensity)
print('unit_u = ', unit_u)
print('unitTime = ', unitTime)
print('unitTime_in_yr = ', unitTime_in_yr)
print('unitTime_in_Myr = ', unitTime_in_Myr)
print('unit_P = ', unit_P)

mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
gamma = 5.0/3.0

mu_bar = 2.29 # For the Boss & Bodenheimer test.

Pconst = kB / mu_bar / mH

print()
print('Pconst = ', Pconst)






