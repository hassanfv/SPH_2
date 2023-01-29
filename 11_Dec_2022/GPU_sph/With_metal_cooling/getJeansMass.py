import numpy as np


RgasConstant = 8.3143e7 # erg.K^-1.mol^-1
M_sun = 1.98992e+33 # gram
G = 6.67259e-8 #  cm3 g-1 s-2
mu = 1.22

Rcld = 0.05 # pc
Rcld_in_cm = Rcld * 3.086e+18 # cm
Mcld_in_g = 0.006 * M_sun

Volume = 4./3. * np.pi * Rcld_in_cm**3

rho = Mcld_in_g / Volume

T = 100. #K

M_J = (5.*RgasConstant*T/2./G/mu)**(3./2.) * (4.*np.pi*rho/3.)**(-1./2.)

print(f'mean density = {rho:.3E}')
print(f'Jeans mass = {M_J/M_sun} MSun')












