
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('HCmu.csv')

# ['lognH', 'rkpc', 'logNHtot', 'logT', 'logHeating', 'logCooling', 'mu']
print(df.columns)

lognH = df['lognH'].values
rkpc = df['rkpc'].values
logNHtot = df['logNHtot'].values
logT = df['logT'].values
logHeating = df['logHeating'].values
logCooling = df['logCooling'].values
mu = df['mu'].values


nx = np.where((lognH == -0.0) & (rkpc == 0.4) & (logNHtot == 20.0))[0]

print(len(nx))

if True:
  plt.scatter(logT[nx], logCooling[nx], s = 20, color = 'blue')
  plt.scatter(logT[nx], logHeating[nx], s = 10, color = 'red')
  plt.show()


