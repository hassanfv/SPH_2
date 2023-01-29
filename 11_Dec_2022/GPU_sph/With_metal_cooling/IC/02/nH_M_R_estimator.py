
import numpy as np

MSun = 1.98892e33

R = 30. # pc
R = R * 3.086e18 # cm
V = 4./3. * np.pi * R*R*R

XH = 0.76
mH = 1.6726e-24 # g
nH = 1e0 # cm^-3
rho = nH * mH / XH # g/cm^3

M_in_g = rho * V

print(f'A cloud with R = {R/3.086e18} pc and nH = {nH} has a mass of {M_in_g/MSun:.3f} M_sun')

