
import numpy as np



kB = 1.3807e-16
mH = 1.673534e-24
gamma = 5./3.

v = 1000. # km/s

T1 = 10000 # K

cs = (gamma * kB * T1 / mH)**0.5 / 100. / 1000.

print(f'Sound speed for T = {T1} K is c_s = {cs:.2f} km/s')

M = v / cs

# Rankine-Hugoniot relation for post-shock temperature
T2_T1_ratio = 2 * gamma * (gamma - 1.) * M**2 / (gamma + 1.)**2

# Calculate post-shock temperature
T2 = T2_T1_ratio * T1

print(f'v = {v} km/s')
print(f'Mach number = {M:.1f}')
print(f'post-shock T = {T2:.1E} K')






