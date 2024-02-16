
import numpy as np
import pandas as pd


#===== find_closest_index
def find_closest_index(X, a):
  differences = [abs(x - a) for x in X]
  closest_index = differences.index(min(differences))
  return closest_index



hplanck = 6.626e-27
clight_in_cm = 3e10 # cm/s
clight_in_A = 3e18 # A/s
ryd_in_Hz = 3.289e15 # Hz

data = pd.read_csv('AGNref.sed', sep='\t', comment='#', header = None)
ryd = data.iloc[:, 0]
nuJnu = data.iloc[:, 1]



nu = ryd * ryd_in_Hz
Jnu = nuJnu / nu # ----> converting to erg/s/Hz

#------> Normalizing to have L912 at Lyman limit <------
L912_A = 3.15e42 # erg/s/A #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
c_nu2 = clight_in_A / ryd_in_Hz / ryd_in_Hz

L912_Hz = c_nu2 * L912_A # erg/s/Hz
nx = find_closest_index(nu, 3.289e15)
print(f'if {nu[nx]:.3E} is close to {ryd_in_Hz:.3E}, then it is fine!')

Jnu = Jnu / Jnu[nx]
Lnu = Jnu * L912_Hz # Setting the value at 912 to L912_Hz ---> Normalizing the SED!

QPhoton = 0.0
for i in range(0, len(nu)-1):
  
  if nu[i] > ryd_in_Hz:
    dnu = nu[i+1] - nu[i]
    QPhoton += dnu * Lnu[i] / hplanck / nu[i]

print(QPhoton)


nH = 1000.0 # cm^-3

dist = 300.0 # pc
dist = dist * 3.086e18 # cm

phi_H = QPhoton / 4.0 / np.pi / dist / dist
logU = np.log10(phi_H / clight_in_cm / nH) # nH = total hydrogen density.

print('logU = ', logU)



Temp = 10000.0
NHtot = 21.30
metallicity = -1.0

lines = [
  f"hden {np.log10(nH):.2f}",
  f"constant temperature {Temp}",
  f"ionization parameter = {logU:.2f}",
  f"AGN T =1.5e5 k, a(ox) = -1.4, a(uv)=-0.5 a(x)=-1",
  f"extinguish column = {NHtot}",
  f"metals {metallicity}",
  f"stop zone 1"
]

with open('script.in', 'w') as f:
  for line in lines:
    f.write(line + '\n')




