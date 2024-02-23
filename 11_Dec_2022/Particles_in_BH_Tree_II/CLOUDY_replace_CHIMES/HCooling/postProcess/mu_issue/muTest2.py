
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


dfHC = pd.read_csv('HCmu.csv')
#dfHC.columns ---> lognH  rkpc  logNHtot  logT  logHeating  logCooling      mu

#print(dfHC)

nH = dfHC['lognH'].values
T = dfHC['logT'].values
rkpc = dfHC['rkpc'].values
NH = dfHC['logNHtot'].values
Heat = dfHC['logHeating'].values
Cool = dfHC['logCooling'].values
mu = dfHC['mu'].values

N_nH = len(nH)
N_T = len(T)
N_r = len(rkpc)
N_NH = len(NH)

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




#--- nH
nxnH0 = -1
for j in range(N_nH):
  if (nxnH0 == -1) & (nH_p <= nH[j]):
    nxnH0 = j

if nxnH0 > 0: # To handle cases in which nH_p <= min(nH)
  nxnH0 = nxnH0 - 1

if nxnH0 == -1: # To handle cases in which nH_p >= max(nH)
  nxnH0 = N_nH - 2 # 1 will be added in nxnH1!

#--- rkpc
nxr0 = -1
for j in range(N_r):
  if (nxr0 == -1) & (r_p <= rkpc[j]):
    nxr0 = j

if nxr0 > 0: # To handle cases in which r_p <= min(r)
  nxr0 = nxr0 - 1

if nxr0 == -1: # To handle cases in which r_p >= max(r)
  nxr0 = N_r - 2

#--- NH
nxNH0 = -1
for j in range(N_NH):
  if (nxNH0 == -1) & (NH_p <= NH[j]):
    nxNH0 = j

if nxNH0 > 0: # To handle cases in which NH_p < min(NH)
  nxNH0 = nxNH0 - 1

if nxNH0 == -1: # To handle cases in which NH_p > max(NH)
  nxNH0 = N_NH - 2


nxnH1 = nxnH0 + 1
nxr1  = nxr0 + 1
nxNH1 = nxNH0 + 1


# Now we isolate those u and mu for which nH, r and NH are nH_p, r_p, and NH_p!
res = []
for i in range(len(Heat)):
  if ((nH[nxnH0] < nH[i] <= nH[nxnH1]) & (rkpc[nxr0] < rkpc[i] <= rkpc[nxr1]) & (NH[nxNH0] < NH[i] <= NH[nxNH1])):
    res.append([T[i], u[i], mu[i]])

res = np.array(res)
df = pd.DataFrame(res)
df.columns = ['T', 'u', 'mu']
df.to_csv('tmp.csv', index = False)

ux = df['u'].values
Tx = df['T'].values

#plt.scatter(df['u'], df['T'], s = 5)
#plt.show()

#--- u
nxT0 = -1
for j in range(len(ux)):
  if (nxT0 == -1) & (u_p <= ux[j]):
    nxT0 = j

if nxT0 > 0: # To handle cases in which T_p <= min(T)
  nxT0 = nxT0 - 1

if nxT0 == -1: # To handle cases in which T_p >= max(T)
  nxT0 = N_T - 2

print(f'u_p = {u_p:3f} corresponds to T = {Tx[nxT0]} for nH_p = {nH_p}, r_p = {r_p}, and NH_p = {NH_p}')






