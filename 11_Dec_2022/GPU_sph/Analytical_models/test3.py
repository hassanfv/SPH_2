
import numpy as np


v_sh = 20000.
nH_pre = 0.2
nH_c_i = 18.

A = (v_sh / 5000.)**(4.2) # v_sh in km/s

B = (nH_pre / 0.02)**2.1 # nH_pre in cm^-3

C = (nH_c_i / 18.)**(-2.1) # nH_c_i in cm^-3

D = A * B * C


NH_c_i = D * 1.4e20


print(f'NH_c_i = {NH_c_i:.2E}')





