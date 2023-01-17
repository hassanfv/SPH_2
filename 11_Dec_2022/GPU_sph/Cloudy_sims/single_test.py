import numpy as np
import pandas as pd


df = pd.read_csv('sorted_CloudyCoolingGrid.csv')

T = df['T'].values
nH = df['nH'].values

N_hden = 351
N_TempArr = 301


TGrid = np.zeros(N_TempArr)
nHGrid = np.zeros(N_hden)

for i in range(N_TempArr):

	TGrid[i] = T[i * N_hden] # # Since for each nH we have the full range of Temp. That is why here it should be N_hden !


for i in range(N_hden):

	nHGrid[i] = nH[i]


#print(TGrid)
#print(nHGrid)


