import numpy as np


kB = 1.3807e-16
mu = 0.61
mH = 1.673534e-24

unit_u = 4100904397311.213
gamma = 5./3.

Temp = 1e10 #(gamma - 1) * mH / kB * mu * u * unit_u

u = 1.0/mu/(gamma - 1.)/mH * kB * Temp


print(f'For T = {Temp} we have u = {u}  or  {u:.4E}')



