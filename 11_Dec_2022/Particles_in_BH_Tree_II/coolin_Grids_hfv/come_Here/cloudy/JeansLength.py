
import numpy as np

X = 0.7
kB = 1.3807e-16
mu = 1.0 #0.61
mH = 1.673534e-24
gamma = 5./3.
G = 6.67430e-8


nH = 10. # cm^-3
T = 10000. # K

cs = np.sqrt(gamma * kB * T / mu / mH)

rho = nH * mH / X

LJeans = np.sqrt(np.pi * cs * cs / G / rho)

print()
print(f'For nH = {nH} and T = {T}, we have LJeans = {LJeans:.4E} cm or {(LJeans/3.086e18):.3f} pc.')
print(f'For this length, we have logNH = {(np.log10(LJeans * nH)):.3f}')
print()

