
import numpy as np

gamma = 5./3.
Mach = 3.

A = (gamma - 1.) * Mach**2 + 2.
B = 2.*gamma*Mach**2 - gamma - 1.

C = (gamma + 1.)**2 * Mach**2

T1 = 54. # K



T2 = A*B/C * T1

print('T2 = ', T2)
