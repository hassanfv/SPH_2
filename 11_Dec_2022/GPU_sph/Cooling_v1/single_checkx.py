
import numpy as np
from photolibs3 import *
import matplotlib.pyplot as plt
import time

XH = 0.76

ref_dt_cgs = 6.31135e+09
current_dt_cgs = 1.0 * ref_dt_cgs
dt = current_dt_cgs

# 3.5453E+12, 3.2363E+12, 3.5318E+12, 2.4828E-24, 2.1209E+10, -3.0897E+11, -8.7150E-02

u_cgs = 1.00E+13
rho_cgs = 1.6726e-23 / 0.76

TA = time.time()
ux = DoCooling_h(rho_cgs, u_cgs, dt, XH)
print('Elapsed time = ', time.time() - TA)

print(f'dt = {dt:.6E}')
print(f'rho = {rho_cgs:.4E}')
print(f'u Before cooling = {u_cgs:.6E}')
print(f'u After cooling = {ux:.6E}')
print(f'delta = {u_cgs-ux:.6E}')


