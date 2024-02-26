
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import pickle


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
    f"save last species densities \"_elc.den\" \"e-\"",
    f"save last heating \"_heating.het\""
  ]
  with open(f'script_{RiD:02}.in', 'w') as f:
    for line in lines:
      f.write(line + '\n')


#-- Reading the file containing the list of parameters. It is a list of list with each sublist containing [lognH[i], logT[j], rkpc[k], logNHtot[p]]
with open('inputLists.pkl', 'rb') as f:
  inputLists = pickle.load(f)
#-----

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


Z = -1.0

RiD = 1

T11 = time.time()

res = []

for i in range(100):

  iList = inputLists[i]
  nH = 10**iList[0]
  T = 10**iList[1]
  dist = iList[2] * 1000.0 # converting kpc to pc!
  logNHtot = iList[3]

  print(f'Modeling lognH = {iList[0]}, logT = {iList[1]}, r = {iList[2]} kpc, logNHtot = {iList[3]} ....')

  createRunScript(nH, logNHtot, T, Z, dist, RiD)

  #---- Executing CLOUDY ----
  command = f"cloudy script_{RiD:02}"
  os.system(command)
  #----

  with open(f"script_{RiD:02}_heating.het", 'r') as file:
    # Read all lines
    lines = file.readlines()
    # Skip the first row
    second_row = lines[1].strip().split('\t')
    # Return the 2nd, 3rd, and 4th numbers from the second row
    numbers = [np.log10(float(second_row[i])) for i in range(1, 4)]  # Assuming numbers are tab-separated
    numbers = [np.round(_, 3) for _ in numbers]
    res.append([np.round(iList[0], 3), np.round(iList[2], 3), np.round(iList[3], 3)] + numbers)

dfHC = pd.DataFrame(res)
dfHC.columns = ['lognH', 'rkpc', 'logNHtot', 'logT', 'logHeating', 'logCooling']

dfHC.to_csv('HCX.csv', index = False)

print(dfHC)
print()

print('Elapsed time = ', time.time() - T11)




