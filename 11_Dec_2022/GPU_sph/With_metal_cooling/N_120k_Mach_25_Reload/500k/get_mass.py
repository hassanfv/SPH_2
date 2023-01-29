
import numpy as np

M_sun = 1.98892e33 # g
mH = 1.6726e-24 # g

XH = 0.76

nH = 710.0 # cm^-3
r_in_pc = 2.3 # pc

r_in_cm = r_in_pc * 3.086e+18

rho = nH / XH * mH

Mass = 4.0/3.0 * np.pi * r_in_cm**3 * rho

print(f'nH = {nH} corresponds to rho = {rho:.3E}')
print(f'Mass in M_sun = {Mass/M_sun:.3E} M_sun')



