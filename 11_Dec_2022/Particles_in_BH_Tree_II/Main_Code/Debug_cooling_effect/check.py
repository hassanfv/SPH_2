
import numpy as no


mu = 0.61
mH = 1.673534e-24
gamma = 5./3.
kB = 1.3807e-16


# rkpc = 0.108, nH = 16.59, ut = 83411584.0000, ut_pre = 83326656.0000, uBad = 31.4086, uAad = 34.7433, uAC = 31.4177

#--- checking from the hdf5 file directly ----
#u before evolution = 6.33550E+13
#u after 3.0040307247699443 yrs of evolution = 5.62136E+13
#---------------------------------------------

unit_u = 1.80046e+12 #!!!!!!!!!!!!!!!!!!!!!!

uBad = 31.4086 * unit_u # Note: uBC = uAad
T_Bad = (gamma - 1) * mH / kB * mu * uBad
print(f'uBad = {uBad:.4E},   T_Bad = {T_Bad:.2E}')
print()

uBC = 34.7433 * unit_u # Note: uBC = uAad
T_BC = (gamma - 1) * mH / kB * mu * uBC
print(f'uBC = {uBC:.4E},   T_BC = {T_BC:.2E}')
print()

uAC = 31.4177 * unit_u
T_AC = (gamma - 1) * mH / kB * mu * uAC
print(f'uAC = {uAC:.4E},   T_AC = {T_AC:.2E}')


