
import numpy as np


#---- Speed of Sound ------
mH = 1.6726e-24 # gram
kB = 1.3807e-16  # cm2 g s-2 K-1
T_0 = 54. # K, see Table_1 in Anathpindika - 2009 - II

# Note that for pure molecular hydrogen mu=2. For molecular gas with ~10% He by mass and trace metals, mu ~ 2.7 is often used.
muu = 2.7
mH2 = muu * mH

c_0 = (kB * T_0 / mH2)**0.5


print(c_0)
