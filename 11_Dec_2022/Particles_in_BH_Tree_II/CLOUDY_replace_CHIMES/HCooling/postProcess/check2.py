
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

nnx = np.where((T < 3) & (nH >= 3))[0]

print(len(nnx))

plt.hist(mu[nnx])
plt.show()
#s()

plt.scatter(T, mu, s = 1)
plt.show()
s()

print('unique rkpc = ', np.unique(rkpc))
print()

nH_unq = np.unique(nH)
rkpc_unq = np.unique(rkpc)
NH_unq = np.unique(NH)

nx_rkpc = 0
nx_nH = 5
nx_NH = -1

nx = np.where((nH == nH_unq[nx_nH]) & (rkpc == rkpc_unq[nx_rkpc]) & (NH == NH_unq[nx_NH]))[0]

C = np.log10(10**Cool[nx] / 10**nH_unq[nx_nH] / 10**nH_unq[nx_nH])
H = np.log10(10**Heat[nx] / 10**nH_unq[nx_nH] / 10**nH_unq[nx_nH])

plt.scatter(T[nx], C, s = 10, color = 'blue')
plt.plot(T[nx], C, color = 'blue')
plt.scatter(T[nx], H, s = 10, color = 'orange')
plt.plot(T[nx], H, color = 'orange')
plt.ylim(-27.5, -21.5)
plt.title(f'nH = {nH_unq[nx_nH]},   rkpc = {rkpc_unq[nx_rkpc]},   NH = {NH_unq[nx_NH]}')

plt.savefig('HC_' + f'nH_{nH_unq[nx_nH]}_rkpc_{rkpc_unq[nx_rkpc]}_NH_{NH_unq[nx_NH]}.png')

plt.show()



