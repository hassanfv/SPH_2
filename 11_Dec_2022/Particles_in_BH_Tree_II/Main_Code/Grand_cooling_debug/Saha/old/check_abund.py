
from photolibs3d import *
import matplotlib.pyplot as plt

XH = 0.76

# u MUST be in physical units !!!!!!
kB = 1.3807e-16  # cm2 g s-2 K-1
mH = 1.6726e-24 # gram
gamma = 5.0/3.0

yHelium = (1.0 - XH) / (4. * XH)

ne_guess = 1.0 # our initial guess is that elec_density = hydrogen density.
gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField()

nHcgs = 0.1
u = 1.7609E+12

temp = 10000

nH0, nHe0, nHp, ne_guess, nHep, nHepp = Abundance_hX(temp, nHcgs, gJH0, gJHe0, gJHep)


print('nH0, nHe0, nHp, ne_guess, nHep, nHepp = ', nH0, nHe0, nHp, ne_guess, nHep, nHepp)

T = np.logspace(2, 9, 100)

nHcgs = 10000.0

res = []

for Temp in T:

  HRate, Lambda, LambdaRecHp, LambdaIonH0 = coolingHeatingRates(Temp, nHcgs)

  res.append([HRate, Lambda, LambdaRecHp, LambdaIonH0])


res = np.array(res)

HRate = res[:, 0]
cRate = res[:, 1]
aRecHp= res[:, 2]
ionRate_H = res[:, 3]

plt.plot(np.log10(T), np.log10(HRate), color = 'k')
plt.plot(np.log10(T), np.log10(cRate), color = 'b')
plt.plot(np.log10(T), np.log10(aRecHp), color = 'r')
plt.plot(np.log10(T), np.log10(ionRate_H), color = 'lime')

plt.xlim(2, 9)
plt.ylim(-25, -21)

plt.show()




