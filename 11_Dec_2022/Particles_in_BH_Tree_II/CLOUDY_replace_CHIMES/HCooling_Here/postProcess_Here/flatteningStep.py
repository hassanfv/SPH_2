
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def closestndx(arr, val):
  return np.argmin(np.abs(arr - val))


dfiF = pd.read_csv('ionFrac.csv')

dfHC = pd.read_csv('HCmu.csv')
#dfHC.columns ---> lognH  rkpc  logNHtot  logT  logHeating  logCooling      mu

print(dfiF)
print()
print(dfHC)

nH = dfHC['lognH'].values
rkpc = dfHC['rkpc'].values
NH = dfHC['logNHtot'].values
T = dfHC['logT'].values
Heat = dfHC['logHeating'].values
Cool = dfHC['logCooling'].values
mu = dfHC['mu'].values

nH = np.round(nH, 3)
rkpc = np.round(rkpc, 3)
NH = np.round(NH, 3)
T = np.round(T, 3)


nHGrid = np.arange(-3, 3.1, 1)
TGrid = np.arange(2, 10.1, 0.1)
rGrid = np.arange(0.1, 0.5, 0.1)
NHGrid = np.arange(19, 23.1, 0.2)

N_nH = len(nHGrid)
N_T = len(TGrid)
N_r = len(rGrid)
N_NH = len(NHGrid)

Gam = np.zeros((N_nH, N_T, N_r, N_NH))
Lam = np.zeros((N_nH, N_T, N_r, N_NH))
Meu = np.zeros((N_nH, N_T, N_r, N_NH))

# Flattening Gam, Lam and Meu
Ntot = N_nH * N_T * N_r * N_NH
GamFlat = np.zeros(Ntot)
LamFlat = np.zeros(Ntot)
MeuFlat = np.zeros(Ntot)

uFlat = kB * T / (gamma - 1.0) / MeuFlat / mH

#---- Preparing 4D array for Gam and Lam!!
for i in range(len(nHGrid)):
  for j in range(len(TGrid)):
    for k in range(len(rGrid)):
      for l in range(len(NHGrid)):
        nt = np.where((nH == np.round(nHGrid[i],3)) & (T == np.round(TGrid[j], 3)) & (rkpc == np.round(rGrid[k],3)) & (NH == np.round(NHGrid[l],3)))[0]
        if len(nt) == 0 or len(nt) > 1:
          print('BADDDD') 
        
        stride_nH = N_T * N_r * N_NH
        stride_T = N_r * N_NH
        stride_r = N_NH
        
        Gam[i, j, k, l] = Heat[nt]
        Lam[i, j, k, l] = Cool[nt]
        Meu[i, j, k, l] = mu[nt]
        
        ndx = i * stride_nH + j * stride_T + k * stride_r + l
        
        GamFlat[ndx] = Heat[nt]
        LamFlat[ndx] = Cool[nt]
        MeuFlat[ndx] = mu[nt]

dictx = {'nH': nHGrid, 'T': TGrid, 'rkpc': rGrid, 'NH': NHGrid, 'Gam': Gam, 'Lam': Lam, 'mu': Meu}
with open('Ready.pkl', 'wb') as f:
  pickle.dump(dictx, f)

# Save arrays and their sizes to a binary file
with open('FlatHCoolMu.bin', 'wb') as f:
    # Write sizes of the arrays
    np.array([len(nHGrid), len(TGrid), len(rGrid), len(NHGrid), len(GamFlat), len(LamFlat), len(MeuFlat)], dtype=np.int32).tofile(f)
    
    # Write the arrays themselves
    nHGrid.tofile(f)
    TGrid.tofile(f)
    rGrid.tofile(f)
    NHGrid.tofile(f)
    GamFlat.tofile(f)
    LamFlat.tofile(f)
    MeuFlat.tofile(f)




