import numpy as np

# ref: Fragile et al - 2004.

ksi = 1 # density contrast (i.e. rho_cl / rho_b).

R_cl = 30 # pc
R_cl = R_cl * 3.086e+18 # cm

v_sh_b = 100 # km/s
v_sh_b = v_sh_b * 1000 * 100 # cm/s

mH = 1.6726e-24 # gram
n_cl = 1.0 # cm^-3
rho_cl = mH * n_cl

v_sh_cl = v_sh_b / np.sqrt(ksi)

t_cc = R_cl / v_sh_cl

C = 7.0e-35 # g cm^-6 s^4
t_cool = C * v_sh_b**3 / ksi**(3./2.) / rho_cl


print(f't_cc = {t_cc/3600/24/365.25:.3f} yrs')
print()
print(f't_cool = {t_cool/3600/24/365.25:.3f} yrs')
print()
print(f't_cool / t_cc = {t_cool / t_cc:.4f}')










