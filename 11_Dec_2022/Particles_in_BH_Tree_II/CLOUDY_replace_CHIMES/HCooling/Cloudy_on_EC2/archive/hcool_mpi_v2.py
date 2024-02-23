
import numpy as np
import pandas as pd
import os
import time
import pickle
from parseIonFrac_v1 import ionFractions
from selectediFrac import selectediFrac
from mpi4py import MPI
import glob



#===== clean_hfv
def clean_hfv(pattern):
  files_to_remove = glob.glob(pattern)
  for file in files_to_remove:
    try:
      os.remove(file)
      print(f"Removed file: {file}")
    except OSError as e:
      print(f"Error: {e.strerror} - {file}")


#===== find_closest_index
def find_closest_index(X, a):
  differences = [abs(x - a) for x in X]
  closest_index = differences.index(min(differences))
  return closest_index


#===== getElectronDensity
def getElectronDensity(RiD):
  with open(f"script_{RiD:02}_elc.den", 'r') as f:
    next(f)
    second_line = f.readline()
    values = second_line.split()
    nelec = float(values[1])
  return nelec


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


#===== mainFunc
def mainFunc(nbeg, nend, RiD):

  resHCmu = []
  resIon = []

  for i in range(nbeg, nend):

    iList = inputLists[i]
    nH = 10**iList[0]
    T = 10**iList[1]
    dist = iList[2] * 1000.0 # converting kpc to pc!
    logNHtot = iList[3]

    print(f'Modeling lognH = {iList[0]}, logT = {iList[1]}, r = {iList[2]} kpc, logNHtot = {iList[3]} ....')

    createRunScript(nH, logNHtot, T, Z, dist, RiD)

    #---- Executing CLOUDY ----
    command = f"/home/ubuntu/cloudy-master/source/cloudy.exe -p script_{RiD:02}"
    os.system(command)
    #----

    with open(f"script_{RiD:02}_heating.het", 'r') as file:
      lines = file.readlines() # Read all lines
      second_row = lines[1].strip().split('\t') # Skip the first row
      # Return the 2nd, 3rd, and 4th numbers from the second row
      numbers = [np.log10(float(second_row[i])) for i in range(1, 4)]  # Assuming numbers are tab-separated
      numbers = [np.round(_, 4) for _ in numbers]

    nelec = getElectronDensity(RiD)
    ionization_file = f"script_{RiD:02}_ionization.dat"
    ionFrac, mu = ionFractions(ionization_file, nelec/nH) #----> iF is similar to ionFrac that I used with CHIMES data!

    resHCmu.append([np.round(iList[0], 4), np.round(iList[2], 4), np.round(iList[3], 4)] + numbers + [np.round(mu, 4)])
    
    iFrac = selectediFrac(ionFrac)
    iFrac = [np.round(np.log10(_), 4) for _ in iFrac]
    
    resIon.append([np.round(iList[0], 4), np.round(iList[1], 4), np.round(iList[2], 4), np.round(iList[3], 4)] + iFrac)

  resHCmu = np.array(resHCmu)
  resIon = np.array(resIon)
  return resHCmu, resIon



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


Z = -1.0 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nCPUs = comm.Get_size()

N = len(inputLists)

#------- used in MPI --------
count = N // nCPUs
remainder = N % nCPUs

if rank < remainder:
	nbeg = rank * (count + 1)
	nend = nbeg + count + 1
else:
	nbeg = rank * count + remainder
	nend = nbeg + count
#--------------------------

RiD = rank

if rank == 0:
	T11 = time.time()
#--------------------------
local_res = mainFunc(nbeg, nend, RiD)
Xres1 = 0 # This is just a placeholder. Its absence would crash this line--> Xres = comm.bcast(Xres, root = 0) for CPUs with rank != 0.
Xres2 = 0

if rank == 0:
	Xres1, Xres2 = local_res
	for i in range(1, nCPUs):
		res_tmp1, res_tmp2 = comm.recv(source = i)
		Xres1 = np.concatenate((Xres1, res_tmp1), axis=0)
		Xres2 = np.concatenate((Xres2, res_tmp2), axis=0)
else:
	comm.send(local_res, dest = 0)

Xres1 = comm.bcast(Xres1, root = 0)
Xres2 = comm.bcast(Xres2, root = 0)
#----------------------------

if rank == 0:
  clean_hfv('script*')
  print('T11 = ', time.time() - T11)

if rank == 0:
  dfHC = pd.DataFrame(Xres1)
  dfHC.columns = ['lognH', 'rkpc', 'logNHtot', 'logT', 'logHeating', 'logCooling', 'mu']
  dfHC.to_csv('HCmu.csv', index = False)
  print(dfHC)
  print()
  
  dfIonFrac = pd.DataFrame(Xres2)
  dfIonFrac.columns = ionID = ["lognH", "logT", "rkpc", "logNHtot", "HI", "H2", "CI", "CII", "CIII", "CIV", "NV",
                               "OI", "OVI", "NaI", "MgI", "MgII", "AlII", "AlIII", "SiII", "SiIII", "SiIV", "PV",
                               "SII", "CrII", "MnII", "FeII", "ZnII"]
  dfIonFrac.to_csv('ionFrac.csv', index = False, float_format='%.4f')

  print(dfIonFrac)




