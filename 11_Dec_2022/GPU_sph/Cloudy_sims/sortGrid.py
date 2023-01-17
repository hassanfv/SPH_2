import numpy as np
import pandas as pd

# NOTE:
# N_hden = 351
# N_TempArr = 301

df = pd.read_csv('CloudyCoolingGridZZZ.csv')
# ['T', 'nH', 'mu', 'heating', 'cooling']

T = df['T'].values
mu = df['mu'].values

u = np.zeros(len(T))

kB = 1.3807e-16  # cm2 g s-2 K-1
mH = 1.6726e-24 # gram
gamma = 5.0/3.0

for i in range(len(T)):

	muT = 1.22

	u[i] = kB/mH/(gamma - 1.0)/muT * T[i]

df['u'] = u

df = df[['u', 'nH', 'mu', 'heating', 'cooling', 'T']]

df.to_csv('CloudyCoolingGrid_TEST.csv', index = False)

df = df.sort_values(by = ['u', 'nH'])
df = df.reset_index(drop = True)

df.to_csv('sorted_CloudyCoolingGrid.csv', index = False)

print(df)
