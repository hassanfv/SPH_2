
import numpy as np


N = 65752 # Number of particles
R = 1.0 # radius of the sphere (in spherical collapse)

d = ((4./3. * np.pi * R**3) / N)**(1./3.)


print(f'Inter-particle separtion in code unit = {np.round(d,4)}')

