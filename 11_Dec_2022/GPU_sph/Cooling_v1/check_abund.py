
from photolibs3 import *

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

mu1 = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess) # yHelium = nHe/nH and ne = ne/nH
temp1 = (gamma - 1.0) * mH / kB * mu1 * u

print('Initial ne_guess = ', ne_guess)
print('Initial mu = ', mu1)
print('Initial Temp = ', temp1)
print()

#temp = temp1

temp = 50000

nH0, nHe0, nHp, ne_guess, nHep, nHepp = Abundance_hX(temp, nHcgs, gJH0, gJHe0, gJHep)

mu = (1.0 + 4. * yHelium) / (1.0 + yHelium + ne_guess) # yHelium = nHe/nH and ne = ne/nH
tempx = (gamma - 1.0) * mH / kB * mu * u


print('nH0, nHe0, nHp, ne_guess, nHep, nHepp = ', nH0, nHe0, nHp, ne_guess, nHep, nHepp)
print('ne_guess = ', ne_guess)
print('mu = ', mu)
print('Temp = ', tempx)


temp = 12000.0
uA = convert_Temp_to_u(temp, nHcgs, XH)

print(f'uA = {uA:.4E}')





