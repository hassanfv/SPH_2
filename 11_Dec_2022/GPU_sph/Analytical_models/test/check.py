import numpy as np


mH = 1.6726e-24
kB = 1.3807e-16
mu = 0.6

n = 60.0
rho_gas = mu * mH * n
rho_e = rho_gas

T = 1e6

E = 3 * kB * rho_e * T / 2. / mu / mH

print(E)
