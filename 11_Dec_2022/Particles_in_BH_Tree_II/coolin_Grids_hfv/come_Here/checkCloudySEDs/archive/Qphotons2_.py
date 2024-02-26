
import numpy as np

hplanck = 6.626e-27
clight = 3e10

#Lnu = 4.28e30 # erg/s/Hz
Llam = 3.25e42 # erg/s/A

ryd_to_Hz = 3.289e15 # 1 Ryd is 3.289e15 Hz

nu1 = 1.0 * ryd_to_Hz # 1 Rydberg
nu2 = 20.0 * ryd_to_Hz # 20 Rydberg

lam1 = clight / nu2  # 45 A ---> Note that lam1 and lam2 are now in cm. To convert to A multiply them by 1e8.
lam2 = clight / nu1  # 912 A

print(lam1, lam2)

s()

grid = np.linspace(lam1, lam2, 1000)

s = 0.0
for i in range(len(grid)-1):
  dlam = grid[i+1] - grid[i]
  s += Llam / hplanck / clight * grid[i] * dlam

print(f'N ionizing photons / s = {s} [s^-1]')


