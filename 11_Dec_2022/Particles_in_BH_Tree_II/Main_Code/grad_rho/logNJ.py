
#ref: Ploeckinger & Schaye - 2020

import numpy as np

nH = 400. # cm^-3
T = 10000. # K

logNJeans = 19.44 + 0.5 * (np.log10(nH) + np.log10(T))

print('logNJeans = ', logNJeans)

