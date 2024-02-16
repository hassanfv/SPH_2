
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


X = df.iloc[:, :4].values
y = logCooling

seed = 10
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# create an xgboost regression model
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

model.fit(X_train, y_train)

yhat = model.predict(X_test)

for i in range(len(X_test)):
  print(yhat[i], y_test[i])


Tgrid = np.linspace(2, 9, 500)
res = []
for i in range(len(Tgrid)):
  res.append([3.0, 0.5, 20.05, Tgrid[i]])

res = np.array(res)
print(res.shape)
ypred = model.predict(res)

nx = np.where((lognH == 3.0) & (rkpc == 0.5) & (logNHtot == 20.0))[0]

print(len(nx))

if True:
  plt.scatter(logT[nx], logCooling[nx], s = 20, color = 'blue')
  plt.scatter(logT[nx], logHeating[nx], s = 10, color = 'red')
  plt.scatter(Tgrid, ypred, s = 5, color = 'lime')
  plt.show()


