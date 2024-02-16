
import numpy as np
import pandas as pd
import os
from parseIonFrac_v1 import ionFractions


#===== find_closest_index
def find_closest_index(X, a):
  differences = [abs(x - a) for x in X]
  closest_index = differences.index(min(differences))
  return closest_index


#===== createRunScript
def createRunScript(nH, logNHtot, Temp, Z, dist, RiD):

  dist = dist * 3.086e18 # cm

  phi_H = QPhoton / 4.0 / np.pi / dist / dist
  logU = np.log10(phi_H / clight_in_cm / nH) # nH = total hydrogen density.

  lines = [
    f"hden {np.log10(nH):.4f} log",
    f"constant temperature {np.log10(Temp):.4} log",
    f"ionization parameter = {logU:.4f}",
    f"AGN T =1.5e5 k, a(ox) = -1.4, a(uv)=-0.5 a(x)=-1",
    f"extinguish column = {logNHtot:.4f}",
    f"metals {Z}",
    f"stop zone 1",
    f"iterate to convergence",
    f"save last ionization means \"_ionization.dat\"",
    f"save last species densities \"_elc.den\" \"e-\""
  ]
  with open(f'script_{RiD:06}.in', 'w') as f:
    for line in lines:
      f.write(line + '\n')


#===== getElectronDensity
def getElectronDensity(RiD):
  with open(f"script_{RiD:06}_elc.den", 'r') as f:
    next(f)
    second_line = f.readline()
    values = second_line.split()
    nelec = float(values[1])
  return nelec



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




nH = 400.0 # cm^-3
dist = 500.0 # pc
Temp = 10000.0
logNHtot = 21.00
Z = -1.0

RiD = 1

createRunScript(nH, logNHtot, Temp, Z, dist, RiD)

#---- Executing CLOUDY ----
command = f"cloudy script_{RiD:06}"
os.system(command)
#----

nelec = getElectronDensity(RiD)

ionization_file = f"script_{RiD:06}_ionization.dat"
ionFrac, mu = ionFractions(ionization_file, nelec/nH) #----> ionFrac is similar to ionFrac that I used with CHIMES data!

print(f'mu = {mu:.4f}')





