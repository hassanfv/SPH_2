
import numpy as np
import pickle


logT = np.arange(2, 11.1, 0.1)
lognH = np.arange(-4, 5.1, 0.1)
rkpc = np.arange(0.02, 0.33, 0.1)
logNHtot = np.arange(16, 23.1, 0.2)

N_T = len(logT)
N_nH = len(lognH)
N_rkpc = len(rkpc)
N_NH = len(logNHtot)

print('rkpc = ', rkpc)
print()

print(f'N_T = {N_T}, N_nH = {N_nH}, N_rkpc = {N_rkpc}, N_NH = {N_NH}')

N_tot = N_T * N_nH * N_rkpc * N_NH

print(f'Number of CHIMES models to be generated = {N_tot}')

res = []

for i in range(N_T):
  for j in range(N_nH):
    for k in range(N_rkpc):
      for l in range(N_NH):
        res.append([logT[i], lognH[j], rkpc[k], logNHtot[l]])

res = np.array(res)
print()
print(res.shape)

with open('inputListsX.pkl', 'wb') as f:
  pickle.dump(res, f)

