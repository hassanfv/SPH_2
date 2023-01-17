
import numpy as np
from photolibs3 import *
import matplotlib.pyplot as plt

T = 10**(4.12)
nHcgs = 0.001

gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep = RadiationField()

#aHp, aHep, aHepp, ad, geH0, geHe0, geHep = RandCIRates(T)
#nH0, nHe0, nHp, ne, nHep, nHepp = Abundance_hX(T, nHcgs, gJH0, gJHe0, gJHep)

nH0, nHe0, nHp, ne, nHep, nHepp = Abundance_hX(T, nHcgs, 0, 0, 0)

print(ne)

#print(gJH0, gJHe0, gJHep)
#print()
#print(HRate_H0, HRate_He0, HRate_Hep)








