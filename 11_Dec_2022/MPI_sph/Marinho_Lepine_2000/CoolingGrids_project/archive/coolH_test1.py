
import numpy as np


T = 1000.
rho = 3e-22 # corresponding to n = 100 cm-3
X = 0.25

K = 2.11 / rho / X * np.exp(-52490./T)

delta = K*K + 4. * K

y1 = (-K + delta**0.5) / 2.
y2 = (-K - delta**0.5) / 2.

mH = 1.6726e-24 # gram


nHI = rho * X / mH * y1

print('nHI = ', nHI)


#print(y1, y2)
#print(delta)
#print(K)
#print('rho for n = 100/cm3 = ', 175.*mH)





