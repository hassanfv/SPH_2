import numpy as np


Tmin = 2.0
Tmax = 10.6

nH_min = -4.
nH_max = 4.0001


#for T in np.arange(Tmin, Tmax, 0.1):
#  print(round(10**T))


for nH in np.arange(nH_min, nH_max, 0.05):
  print(round(10**nH))
