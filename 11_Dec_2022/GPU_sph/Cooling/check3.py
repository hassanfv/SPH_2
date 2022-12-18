
import numpy as np

from photolibs import *

import matplotlib.pyplot as plt


nHcgs = 1e0

Tgrid = np.logspace(4, 8, 100)

res = []

for T in Tgrid:

	Heat, Cool = coolingHeatingRates(T, nHcgs)
	
	res.append([Heat, Cool])


res = np.array(res)

Heat = res[:, 0]
Cool = res[:, 1]

plt.plot(np.log10(Tgrid), np.log10(Cool))
plt.plot(np.log10(Tgrid), np.log10(Heat))

plt.xlim(4, 7)
plt.ylim(-24.5, -21.5)


plt.show()

