
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dfHC = pd.read_csv('HCmu.csv')
#dfHC.columns ---> lognH  rkpc  logNHtot  logT  logHeating  logCooling      mu

print()
print(dfHC)

nH = dfHC['lognH'].values
rkpc = dfHC['rkpc'].values
NH = dfHC['logNHtot'].values
T = dfHC['logT'].values
Heat = dfHC['logHeating'].values # Note that the way they are created made them to also be flat.. so use stride!!! GREAT!!
Cool = dfHC['logCooling'].values
mu = dfHC['mu'].values

nH = np.round(nH, 3)
rkpc = np.round(rkpc, 3)
NH = np.round(NH, 3)
T = np.round(T, 3)

nHGrid = np.arange(-4, 4.1, 0.1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TGrid = np.arange(2, 11.1, 0.1)  #!!!!!!!!!!!!! Should be similar to what is in createInputList.py !!!!!!!!!!!!!!!!
rGrid = np.arange(0.1, 0.31, 0.1)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NHGrid = np.arange(16, 23.1, 0.2)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N_nH = len(nHGrid)
N_T = len(TGrid)
N_r = len(rGrid)
N_NH = len(NHGrid)

kB = 1.3807e-16
mH = 1.673534e-24
gamma = 5./3.

uFlat = kB * 10**T / (gamma - 1.0) / mu / mH
uFlat = np.log10(uFlat)
uFlat = np.round(uFlat, 3)
print(np.sort(uFlat))

dictx = {'nH': nHGrid, 'T': TGrid, 'rkpc': rGrid, 'NH': NHGrid, 'Gam': Heat, 'Lam': Cool, 'mu': mu}
with open('Ready.pkl', 'wb') as f:
  pickle.dump(dictx, f)

#------> Writing to be read in C++ <--------
# Convert scalar values to numpy arrays with a single element for uniform handling, using float32 for compatibility with C++ float
N_nH_array = np.array([N_nH], dtype=np.int32)
N_T_array = np.array([N_T], dtype=np.int32)
N_r_array = np.array([N_r], dtype=np.int32)
N_NH_array = np.array([N_NH], dtype=np.int32)

# Ensure all other arrays are of type float32
nHGrid = nHGrid.astype(np.float32)
TGrid = TGrid.astype(np.float32)
rGrid = rGrid.astype(np.float32)
NHGrid = NHGrid.astype(np.float32)
Heat = Heat.astype(np.float32)
Cool = Cool.astype(np.float32)
mu = mu.astype(np.float32)
uFlat = uFlat.astype(np.float32)


# Write to binary file
with open('HCoolMu.bin', 'wb') as bin_file:
    N_nH_array.tofile(bin_file)
    N_T_array.tofile(bin_file)
    N_r_array.tofile(bin_file)
    N_NH_array.tofile(bin_file)
    
    nHGrid.tofile(bin_file)
    TGrid.tofile(bin_file)
    rGrid.tofile(bin_file)
    NHGrid.tofile(bin_file)
    Heat.tofile(bin_file)
    Cool.tofile(bin_file)
    mu.tofile(bin_file)
    uFlat.tofile(bin_file)


