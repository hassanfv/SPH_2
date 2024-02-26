
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


def predict_mu(nH, u, rkpc, NH, coefficients):

    a0, a1, a2, a3, a4, a5, a6, a7, a8 = coefficients
    mu = a0 + a1*nH + a2*u + a3*rkpc + a4*NH + a5*nH**2 + a6*u**2 + a7*rkpc**2 + a8*NH**2
    return mu


dfHC = pd.read_csv('HCmu.csv')
#dfHC.columns ---> lognH  rkpc  logNHtot  logT  logHeating  logCooling      mu

print(dfHC)

nH = dfHC['lognH'].values
T = dfHC['logT'].values
rkpc = dfHC['rkpc'].values
NH = dfHC['logNHtot'].values
Heat = dfHC['logHeating'].values
Cool = dfHC['logCooling'].values
mu = dfHC['mu'].values

kB = 1.3807e-16
mH = 1.673534e-24
gamma = 5./3.

u = kB * 10**T / (gamma - 1.) / mu / mH
u = np.log10(u)

# Prepare the design matrix with the given polynomial terms
X = np.column_stack((np.ones(nH.shape), nH, T, rkpc, NH, nH**2, T**2, rkpc**2, NH**2))

# Initialize and fit the Linear Regression model
model = LinearRegression().fit(X, mu)

# The coefficients (including the intercept)
coefficients = model.coef_
intercept = model.intercept_

# Adjusting the first coefficient to include the intercept
coefficients[0] = intercept

print("Coefficients:", coefficients)
print()

'''
nH0 = 3.1
T0 = 2.7
rkpc0 = 0.2
NH0 = 20.1
'''

nH0, rkpc0, NH0, T0, mu0 = 3.0,0.4,20.6,3.0,0.9319

u0 = kB * 10**T0 / (gamma - 1.) / mu0 / mH
u0 = np.log10(u0)

print(u0)


mux = predict_mu(nH0, u0, rkpc0, NH0, coefficients)

print(mux)



