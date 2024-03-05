
import numpy as np
import pickle


TGrid = np.arange(2, 11.1, 0.1)
nHGrid = np.arange(-4, 5.1, 0.1)
rGrid = np.arange(0.02, 0.73, 0.1)
NHGrid = np.arange(16, 23.1, 0.2)

N_T = len(TGrid)
N_nH = len(nHGrid)
N_rGrid = len(rGrid)
N_NH = len(NHGrid)

print('rGrid = ', rGrid)
print()

print(f'N_T = {N_T}, N_nH = {N_nH}, N_rGrid = {N_rGrid}, N_NH = {N_NH}')

N_tot = N_T * N_nH * N_rGrid * N_NH

print(f'Number of CHIMES models to be generated = {N_tot}')

res = []

for i in range(N_T):
  for j in range(N_nH):
    for k in range(N_rGrid):
      for l in range(N_NH):
        res.append([TGrid[i], nHGrid[j], rGrid[k], NHGrid[l]])

res = np.array(res)
print()
print(res.shape)

with open('inputListsX.pkl', 'wb') as f:
  pickle.dump(res, f)

