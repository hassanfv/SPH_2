
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import time


dfHC = pd.read_csv('../HCmu.csv')
#dfHC.columns ---> lognH  rkpc  logNHtot  logT  logHeating  logCooling      mu

#print(dfHC)

nH = dfHC['lognH'].values
T = dfHC['logT'].values
rkpc = dfHC['rkpc'].values
NH = dfHC['logNHtot'].values
Heat = dfHC['logHeating'].values
Cool = dfHC['logCooling'].values
mu = dfHC['mu'].values

nHGrid = np.arange(-4, 4.1, 0.1)
TGrid = np.arange(2, 11.1, 0.1)
rGrid = np.arange(0.1, 0.31, 0.1)
NHGrid = np.arange(16, 23.1, 0.2)

N_nH = len(nHGrid)
N_T = len(TGrid)
N_r = len(rGrid)
N_NH = len(NHGrid)

kB = 1.3807e-16
mH = 1.673534e-24
gamma = 5./3.

u = kB * 10**T / (gamma - 1.) / mu / mH
u = np.round(np.log10(u), 2)

#u = np.unique(np.round(u, 2))
#print('len(uniq(u)) = ', len(u))


nH_p, r_p, NH_p, T_p, mu_p = 3.0,0.3,22.8,4.1,0.8181

u_p = kB * 10**T_p / (gamma - 1.) / mu_p / mH
u_p = np.log10(u_p)


T1 = time.time()


T_nH = time.time()
#--- nH
nxnH0 = -1
for j in range(N_nH):
  if (nxnH0 == -1) & (nH_p <= nHGrid[j]):
    nxnH0 = j

if nxnH0 > 0: # To handle cases in which nH_p <= min(nH)
  nxnH0 = nxnH0 - 1

if nxnH0 == -1: # To handle cases in which nH_p >= max(nH)
  nxnH0 = N_nH - 2 # 1 will be added in nxnH1!

print('Ttime nH = ', time.time() - T_nH)


#--- rkpc
nxr0 = -1
for j in range(N_r):
  if (nxr0 == -1) & (r_p <= rGrid[j]):
    nxr0 = j

if nxr0 > 0: # To handle cases in which r_p <= min(r)
  nxr0 = nxr0 - 1

if nxr0 == -1: # To handle cases in which r_p >= max(r)
  nxr0 = N_r - 2

#--- NH
nxNH0 = -1
for j in range(N_NH):
  if (nxNH0 == -1) & (NH_p <= NHGrid[j]):
    nxNH0 = j

if nxNH0 > 0: # To handle cases in which NH_p < min(NH)
  nxNH0 = nxNH0 - 1

if nxNH0 == -1: # To handle cases in which NH_p > max(NH)
  nxNH0 = N_NH - 2


nxnH1 = nxnH0 + 1
nxr1  = nxr0 + 1
nxNH1 = nxNH0 + 1

print()
print('Elapsed time (T1) = ', time.time() - T1)

T2 = time.time()

# Now we isolate those u and mu for which nH, r and NH are nH_p, r_p, and NH_p!
res = []
nxT0 = -1
for i in range(len(Heat)):
  if ((nHGrid[nxnH0] < nH[i] <= nHGrid[nxnH1]) & (rGrid[nxr0] < rkpc[i] <= rGrid[nxr1]) & (NHGrid[nxNH0] < NH[i] <= NHGrid[nxNH1])):
    if (nxT0 == -1) & (u_p <= u[i]):
      nxT0 = i

if nxT0 > 0: # To handle cases in which T_p <= min(T)
  nxT0 = nxT0 - 1

if nxT0 == -1: # To handle cases in which T_p >= max(T)
  nxT0 = N_T - 2

print(f'u_p = {u_p:3f} corresponds to T = {T[nxT0]} for nH_p = {nH_p}, r_p = {r_p}, and NH_p = {NH_p}')

print()
print('Elapsed time (T2) = ', time.time() - T2)




