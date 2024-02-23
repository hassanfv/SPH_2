
import numpy as np
import pickle


lognH = np.arange(-4, 4.1, 0.1)
logT = np.arange(2, 11.1, 0.1)
rkpc = np.arange(0.1, 0.31, 0.1)
logNHtot = np.arange(16, 23.1, 0.2)

N_nH = len(lognH)
N_T = len(logT)
N_rkpc = len(rkpc)
N_NH = len(logNHtot)

print(f'N_nH = {N_nH}, N_T = {N_T}, N_rkpc = {N_rkpc}, N_NH = {N_NH}')

N_tot = N_nH * N_T * N_rkpc * N_NH

print(f'Number of CLOUDY models to be generated = {N_tot}')

res = []

for i in range(N_nH):
  for j in range(N_T):
    for k in range(N_rkpc):
      for p in range(N_NH):
        res.append([lognH[i], logT[j], rkpc[k], logNHtot[p]])

res = np.array(res)
print()
print(res.shape)

with open('inputLists.pkl', 'wb') as f:
  pickle.dump(res, f)

