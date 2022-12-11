
import numpy as np

M_sun = 1.989e33 # gram
G_cgs = 6.67259e-8 #  cm3 g-1 s-2

R = 5e16 # cm
M = 1. * M_sun

unitVelocity = (G_cgs * M / R)**0.5
unitDensity = 3.*M/4./np.pi/R**3
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


#------------------

gamma = 5./3.
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

mu = 2. # for a gas of pure molecular hydrogen.

T = 10. # Kelvin

u = kB * T / mu / mH / (gamma - 1.)

print('u (in physical units) = ', u)
print('u (in code units) = ', u/unit_u)


#---- calculating the Jeans Mass -----
R_gas_in_cgs = 8.314e7 # erg/mol/K
T_gas_in_K = 4. # K
mu_H2 = 2.
rho = 1.4e-17
G_cgs = 6.67259e-8 #  cm3 g-1 s-2

MJ = (np.pi * R_gas_in_cgs * T_gas_in_K / G_cgs / mu_H2)**(3./2.) / (rho)**0.5

print('Jeans Mass in code unit = ', MJ/M_sun)



#------ From Hubber et al - 2011 ----------
T_0 = 10. # K
rho_crit = 1e-13 #g/cm3
T = T_0 * rho * (1. + (rho/rho_crit)**(gamma - 1.))

print('T from Hubber = ', T)








