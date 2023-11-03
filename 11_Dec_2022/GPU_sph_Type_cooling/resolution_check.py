
import numpy as np


Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
mH = 1.6726219e-24  # Proton mass in grams

M_cloud = 1e5 # M_sun
N_particles = 13366240

m_p = M_cloud / N_particles

print()


csound = 250000 # cm/s

rho_max = 5e-12

MJ = np.pi**(5/2.) / 6. * csound**3 / np.sqrt(G**3 * rho_max)

MJ_in_Msun = MJ/Msun

print(f'MJ_min = {MJ_in_Msun:.5f} M_sun')
print()

Nngb = 40

m_r = MJ_in_Msun / (2. * Nngb)

print(f'm_p = {m_p:.4E} M_sun')
print(f'm_r = {m_r:.3E} M_sun')

print()
print('If m_p is smaller than or equal to m_r then the resolution requirement is satisfied !!')
