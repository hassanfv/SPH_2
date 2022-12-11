
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dfG = pd.read_csv('div_curlV_from_cpp_130k.csv')
divG = dfG['divV'].values
curlG= dfG['curlV'].values


dfC = pd.read_csv('div_curlV.csv')
divC = dfC['divV'].values
curlC= dfC['curlV'].values


#Y = curlG - curlC
Y = divG - divC
X = np.arange(len(Y))

Y = Y[::50]
X = X[::50]

plt.plot(X, Y, color = 'k')

plt.show()
