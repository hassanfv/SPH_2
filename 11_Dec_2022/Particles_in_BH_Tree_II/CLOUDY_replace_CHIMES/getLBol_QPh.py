import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#===== find_closest_index
def find_closest_index(X, a):
    # Compute the absolute difference between each element in X and a
    differences = [abs(x - a) for x in X]
    
    # Find the index of the smallest difference
    closest_index = differences.index(min(differences))
    
    return closest_index


hplanck = 6.626e-27
clight_in_A = 3e18 #--> clight in A/s. It must be in Angstrom/s so that units and dimensions are correct!!!


# Read the file into a DataFrame, assuming it's tab-separated
data = pd.read_csv('zzzz.sed', sep='\t', comment='#', header = None)

print(data.head())

x = data.iloc[:, 0]  # First column as independent variable ====> Rydberg
y1 = data.iloc[:, 1]  # Second column as first dependent variable ===> 4*pi*nu*Jnu

ryd_to_Hz = 3.289e15 # 1 Ryd is 3.289e15 Hz

x_in_Hz = x * ryd_to_Hz
yy = y1 / x_in_Hz

#-------
# Our goal is to have total, i.e. Bolometric luminosity of 1e46 erg/s, as an example!
# We vary the L912 until the integration equals 1e46! L912 is the luminosity at Lyman Limit!
L912_A = 3.15e42 # erg/s/A
c_nu2 = clight_in_A / ryd_to_Hz / ryd_to_Hz

L912_Hz = c_nu2 * L912_A # erg/s/Hz
nx = find_closest_index(x_in_Hz, 3.289e15)
print(f'{x_in_Hz[nx]:.3E}')

yy = yy / yy[nx]
yy = yy * L912_Hz

NPhoton = 0.0
LBol = 0.0
for i in range(0, len(x_in_Hz)-1):
  
  if x_in_Hz[i] > ryd_to_Hz:
    dnu = x_in_Hz[i+1] - x_in_Hz[i]
    NPhoton += dnu * yy[i] / hplanck / x_in_Hz[i]
    
  LBol += yy[i] * x_in_Hz[i]


print()
print(f'for L912_A = {L912_A:.3E} erg/s/A, number of H-ionizing photons = {NPhoton:.3E},  LBol = {LBol:.3E}')
print()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
#plt.loglog(x, y1, label='Incident radiation')
plt.scatter(x, y1, label='Incident radiation')
plt.xlabel('E [Ryd]')
plt.ylabel('Incident radiation')
plt.xscale('log')
plt.yscale('log')
#plt.xlim(1e-4, 1e6)
#plt.ylim(1, 1e9)
plt.legend()


plt.subplot(1,2,2)
plt.loglog(x_in_Hz, 4*np.pi*x_in_Hz*yy)


plt.tight_layout()


plt.savefig('sed_Extinguished.png')

plt.show()



