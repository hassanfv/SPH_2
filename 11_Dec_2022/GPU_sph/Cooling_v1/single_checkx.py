
import numpy as np
from photolibs3 import *
import matplotlib.pyplot as plt
import time

XH = 0.76

ref_dt_cgs = 6.31135e+09
current_dt_cgs = 1.0 * ref_dt_cgs
dt = current_dt_cgs

#0.0642, 1.6936E+12, 1.6878E+12, 1.6936E+12, 2.2175E-25, 2.3071E+07, -5.8585E+09
#0.0657, 1.6878E+12, 1.6827E+12, 1.6878E+12, 2.2175E-25, 2.3486E+07, -5.1150E+09
#0.0673, 1.6827E+12, 1.6776E+12, 1.6827E+12, 2.2176E-25, 2.3906E+07, -5.0728E+09

#1.6548E+12,2.3302E-25,6.3114E+09,5.1418E+09
#1.6548E+12,2.3626E-25,6.3114E+09,5.4357E+09


u_cgs = 1.6548E+12
rho_cgs = 2.3626E-25

TA = time.time()
ux = DoCooling_h(rho_cgs, u_cgs, dt, XH)
print('Elapsed time = ', time.time() - TA)

print(f'dt = {dt:.6E}')
print(f'rho = {rho_cgs:.4E}')
print(f'u Before cooling = {u_cgs:.6E}')
print(f'u After cooling = {ux:.6E}')
print(f'delta = {u_cgs-ux:.6E}')


