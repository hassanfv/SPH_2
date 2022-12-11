
import numpy as np



M_sun = 1.989e33 # gram
G = 6.67259e-8 #  cm3 g-1 s-2

#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1

rho = 1.3e-20 # g/cm^2
Tcld = 10. # K, 

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.7
mH2 = muu * mH

c_s = (kB * Tcld / mH2)**0.5

print('Sound speed = ', c_s)
#--------------------------


M_J = np.pi**2.5 * c_s**3 / 6. / G**1.5 / rho**0.5


print()
print('M_J (M_sun) = ', M_J/M_sun)

m = 5.06e-5 * 50.0 * M_sun

N_ngb = 55.

M_min = N_ngb * m

print()
print('M_min (M_sun) = ', M_min/M_sun)






