import numpy as np


M_tot_in_Msun = 4.18543e+07 # MSun #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Need to be change for each simulation !!!!!!!!!!!!!!!!!
mSPH = 2.3892400e-07 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Need to be change for each simulation !!!!!!!!!!!!!!!!!

Msun = 1.989e33 # g
mp = mSPH * M_tot_in_Msun * Msun

kB = 1.3807e-16
gamma = 5./3.
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2

m = mH = 1.673534e-24

T = 100. # K #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
nH = 1000. # cm^-3 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


rho = nH * mH

cs = (gamma * kB * T / m)**0.5

MJ = np.pi**(5./2.) / 6. * cs**3 / ((G**3 * rho)**0.5)

Nngb = 64

mr = MJ / 2 / Nngb

print(mp, mr) # mp should be smaller than mr!

#----- Estimating some floor for u to set as the lower limit of u in the cooling function
mu = 0.61 # Since in regions with low T, mu can be higher than 0.61, therefore, mu=0.61 gives save value of lower limit for u
T = 100 # K

u = kB * T / mu / mH / (gamma - 1.)
print()
print()
print(f'set u = {u:.4E} as the lower limit in the cooling function')

