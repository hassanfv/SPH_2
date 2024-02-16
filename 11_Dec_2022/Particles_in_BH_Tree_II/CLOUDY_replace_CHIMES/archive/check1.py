
import numpy as np
import pandas as pd


#===== find_closest_index
def find_closest_index(X, a):
  differences = [abs(x - a) for x in X]
  closest_index = differences.index(min(differences))
  return closest_index



hplanck = 6.626e-27
clight_in_A = 3e18 #--> clight in A/s. It must be in Angstrom/s so that units and dimensions are correct!!!


# Read tab-separated file
data = pd.read_csv('AGNref.sed', sep='\t', comment='#', header = None)

x = data.iloc[:, 0]  # First column ====> Rydberg
y1 = data.iloc[:, 1]  # Second column ===> 4*pi*nu*Jnu in erg/s ---> Note that it will be normalized below at Lyman Limit!

ryd_to_Hz = 3.289e15 # 1 Ryd is 3.289e15 Hz

x_in_Hz = x * ryd_to_Hz
yy = y1 / x_in_Hz # ----> converting to erg/s/Hz

#------> Normalizing to have L912 at Lyman limit <------
L912_A = 3.15e42 # erg/s/A #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
c_nu2 = clight_in_A / ryd_to_Hz / ryd_to_Hz

L912_Hz = c_nu2 * L912_A # erg/s/Hz
nx = find_closest_index(x_in_Hz, 3.289e15)
print(f'if {x_in_Hz[nx]:.3E} is close to {ryd_to_Hz:.3E}, then it is fine!')

tmp = yy[nx]
yy = yy / tmp
yy = yy * L912_Hz # Setting the value at 912 to L912_Hz.

NPhoton = 0.0
for i in range(0, len(x_in_Hz)-1):
  
  if x_in_Hz[i] > ryd_to_Hz:
    dnu = x_in_Hz[i+1] - x_in_Hz[i]
    NPhoton += dnu * yy[i] / hplanck / x_in_Hz[i]

print(NPhoton)


