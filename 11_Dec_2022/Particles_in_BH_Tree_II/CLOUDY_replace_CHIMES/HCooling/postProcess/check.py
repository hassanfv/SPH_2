
import numpy as np
import pandas as pd

dfiF = pd.read_csv('ionFrac.csv')
dfHC = pd.read_csv('HCmu.csv')

print(dfiF)

print()

print(dfHC)

rkpc = dfiF['rkpc'].values

print(np.unique(rkpc))

print()

