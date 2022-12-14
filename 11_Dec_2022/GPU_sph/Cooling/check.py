
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#===== RandCIRates (Recombination and Collisional Ionization Rates)
def RandCIRates(T): # T in Kelvin.

	# ****** Recombination and Collisional Ionization Rates ************
	Tfact = 1.0 / (1.0 + np.sqrt(T/1e5))

	# recombination (Cen 1992):
	# Hydrogen II:
	AlphaHp = 8.41e-11 * (T/1000.0)**(-0.2) / (1. + (T/1e6)**(0.7)) / np.sqrt(T)
	# Helium II:
	AlphaHep = 1.5e-10 * T**(-0.6353)
	# Helium III:
	AlphaHepp = 4. * AlphaHp

	# dielectric recombination
	Alphad = 1.9e-3 * T**(-1.5) * np.exp(-470000.0/T) * (1. + 0.3 * np.exp(-94000.0/T))

	# collisional ionization (Cen 1992):
	# Hydrogen:
	GammaeH0   = 5.85e-11 * np.sqrt(T) * np.exp(-157809.1/T) * Tfact
	# Helium:
	GammaeHe0  = 2.38e-11 * np.sqrt(T) * np.exp(-285335.4/T) * Tfact
	# Helium II:
	GammaeHep  = 5.68e-12 * np.sqrt(T) * np.exp(-631515.0/T) * Tfact
	#*******************************************************************

	return AlphaHp, AlphaHep, AlphaHepp, Alphad, GammaeH0, GammaeHe0, GammaeHep


Tgrid = np.logspace(4, 8, 100)

res = []

for T in Tgrid:

	aHp, aHep, aHepp, ad, geH0, geHe0, geHep = RandCIRates(T)

	res.append([aHp, aHep, aHepp, ad, geH0, geHe0, geHep])


res = np.array(res)

aHp   = res[:, 0]
aHep  = res[:, 1]
aHepp = res[:, 2]
ad    = res[:, 3]
geH0  = res[:, 4]
geHe0 = res[:, 5]
geHep = res[:, 6]

#plt.plot(np.log10(Tgrid), np.log10(aHp))

plt.plot(Tgrid, aHp, color = 'black', label = 'aHp')
plt.plot(Tgrid, aHep, color = 'blue', label = 'aHep')
plt.plot(Tgrid, aHepp, color = 'lime', label = 'aHepp')

plt.plot(Tgrid, geH0, color = 'purple', label = 'geH0')
plt.plot(Tgrid, geHe0, color = 'pink', label = 'geHe0')
plt.plot(Tgrid, geHep, color = 'orange', label = 'geHep')

plt.yscale('log')
plt.xscale('log')

plt.legend()
plt.show()








