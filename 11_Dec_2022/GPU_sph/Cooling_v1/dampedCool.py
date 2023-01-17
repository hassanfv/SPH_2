
import numpy as np
from photolibs3 import *
import matplotlib.pyplot as plt
import time

XH = 0.76

ref_dt_cgs = 6.31135e+09
current_dt_cgs = 1.0 * ref_dt_cgs
dt = current_dt_cgs

u_cgs = 1.00E+13
rho_cgs = 1.6726e-22 / 0.76

u = 1.00E+13

dudt_rad = -9.50127e+12 / dt
dudt_ad =  -9.50127e+12 / dt

a = 0.5 * u / dt + dudt_ad

dudt_damped = a * dudt_rad / np.sqrt(a*a + (dudt_rad)**2)


uAf = u + (dudt_ad + dudt_rad) * dt

uAf_damped = u + dudt_ad * dt + dudt_damped * dt

print(f'u Before cooling = {u:.6E}')
print(f'u After cooling = {uAf:.6E}')
print(f'u After cooling (damped) = {uAf_damped:.6E}')


