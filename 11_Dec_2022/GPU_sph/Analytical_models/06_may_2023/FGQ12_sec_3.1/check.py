
import numpy as np


L_AGN = 1e46 # erg/s
Tou_in = 1.0
R_s = 10. * 3.086e+18 # pc to cm
v_s = 1000. * 1000. * 100. # cm/s
v_in = 30000. * 1000. * 100. # cm/s
clight = 29979245800. # cm/s
mH = 1.6726e-24


n_p = Tou_in * L_AGN / np.pi / R_s / R_s / v_s / v_in / clight / mH


print(f'np = {n_p}')


