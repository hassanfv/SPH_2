
import numpy as np


N = 20

s = 0.
for j in range(0, N, 2):

	s += (2.*j+1.) * np.exp(-j*(j+1.))

print(f'z_p = {s}')




s = 0.
for j in range(1, N, 2):

	s += (2.*j+1.) * np.exp(-j*(j+1.))

print(f'z_o = {s}')
