import numpy as np


G = 6.6738e-08
#c_s =  822571.580

mH = 1.6726e-24
kB = 1.3807e-16
XH = 0.76
muu = 1.22 # gas with atomic hydrogen + helium

T = 500

c_s = (kB * T / muu/mH)**0.5

print(f'c_s = ', c_s)

rho = 1.67e-24 * 100

M_J = np.pi ** (5./2.) / 6.0 * c_s**3 / np.sqrt(G**3 * rho)


print(f'M_J = {M_J}')
