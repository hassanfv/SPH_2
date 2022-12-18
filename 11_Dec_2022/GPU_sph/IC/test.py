
import numpy as np

MSun = 1.98892e33

R = 10. # pc
R = R * 3.086e18 # cm
V = 4./3. * np.pi * R*R*R


mH = 1.6726e-24 # g
nH = 1.0 # cm^-3
rho = nH * mH # g/cm^3


M_in_g = rho * V

print(f'M_in_g = {M_in_g:.3E} g')
print(f'M = {M_in_g/MSun:.3f} M_sun')


