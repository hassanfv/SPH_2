import numpy as np


Q = 6.8e55

nH = 400. #cm^-3

r = 0.2 #kpc
r = r * 3.086e21 #cm
c = 29970254700. #cm/s

U = Q / 4. / np.pi / r / r / nH / c

print(f'logU = {(np.log10(U)):.3f}')


