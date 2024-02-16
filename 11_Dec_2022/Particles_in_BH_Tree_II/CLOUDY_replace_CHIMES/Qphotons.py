
import numpy as np

hplanck = 6.626e-27
clight = 3e18 #--> clight in A/s. It must be in Angstrom/s so that units and dimensions are correct!!!

Llam = 3.15e42 # erg/s/A #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ryd_to_Hz = 3.289e15 # 1 Ryd is 3.289e15 Hz

nu1 = 1.0 * ryd_to_Hz # 1 Rydberg
nu2 = 20.0 * ryd_to_Hz # 20 Rydberg

lam1 = clight / nu2 # 45  Angstrom ---> equivalent to 20 Ryd
lam2 = clight / nu1 # 912 Angstrom ---> equivalent to 1 Ryd

print(f'lam1 = {lam1:.2f} A,  lam2 = {lam2:.2f} A')
print()

grid = np.linspace(lam1, lam2, 1000)

Llam = np.zeros_like(grid) + Llam

s = 0.0
for i in range(len(grid)-1):
  dlam = grid[i+1] - grid[i]
  s += 1.0/hplanck/clight * grid[i] * Llam[i] * dlam

print(f'N ionizing photons/s = {s:.3E} [s^-1]')


