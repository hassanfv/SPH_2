import numpy as np


# ref: Arata et al - 2018

gamma = 5./3.
mu = 1.5 # for gas with ionized hydrogen and "partially" ionized Helium gas
kB = 1.3807e-16
mH = 1.6726e-24

#===== csound
def csound(T):
  return (gamma * kB * T / mu / mH)**0.5


T = 8000.0 # K
Z_Z0 = 0.1 # Metallicity relative to the solar.
nH = 10 # cm^-3

Gamma = 2e-26 # erg/s

Lambda = 1.4e-2 * T**0.5 * np.exp(-92./T) * Gamma * Z_Z0 # erg.s^-1.cm^3

cs = csound(T)


tcool = (3./2. * nH * kB * T) / (nH * nH * Lambda)

print('tcool = ', tcool / 3600/24/365.25/1e6)

l_ac = cs * tcool

pc_in_cm = 3.086e+18

print(l_ac/pc_in_cm)



