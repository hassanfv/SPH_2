
# Here we can check if the jeans criteria is satisfied given Nparticle, Mcld, Tcld, rho, rho_crit, ... etc.

import numpy as np

#--- These must be updated for new systems ----
#--- Also the EOS below must be appropriately modified for the new systems ----
rho = 5.0e-12 # g/cm^3
N_neigh = 50
Npart = 13366240
Mcld_in_M_sun = 1.0e5
rho_crit = 5.0e-14 # g/cm^3 
Tcld = 100.
muu = 2.40
#-----------------------------------------------

M_sun = 1.98992e+33 # gram
G = 6.67259e-8 #  cm3 g-1 s-2
gamma = 5./3.
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

mH2 = muu * mH

c_0 = (kB * Tcld / mH2)**0.5
print(f'Isothermal speed of sound = {c_0:.2f} cm/s')

c_s = (c_0*c_0 * (1. + (rho/rho_crit)**(gamma-1)))**0.5  #!!!!!!! # This sound speed will be used to get M_J NOT c_0!
print(f'c_s = {c_s:.2f}')

mpart = Mcld_in_M_sun / Npart # the mass of the SPH particles

print(f'mass of a single SPH particle is {mpart:.3E} M_sun')

M_J = np.pi**(5./2.) / 6. * c_s**3 / np.sqrt(G**3*rho)
M_J_in_M_sun = M_J/M_sun

print(f'Jeans mass at T = {Tcld} K and rho = {rho:.3E} g/cm^3 is {M_J_in_M_sun:.3E} M_sun')

mr = M_J_in_M_sun / (2.*N_neigh)

print(f'Minimum resolvable mass is {mr:.3E} M_sun')

crit = mpart/mr

print()
if crit <= 1.0:
	print('*********************************************************')
	print(f'GREAT !!! Jeans condition is satisfied ! (mp/mr = {crit:.3f})')
	print('*********************************************************')
else:
	print('*********************************************************')
	print(f'SORRY !!! Jeans condition is NOT satisfied ! (mp/mr = {crit:.3f})')
	print('*********************************************************')



