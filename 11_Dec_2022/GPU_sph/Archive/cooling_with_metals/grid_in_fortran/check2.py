import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('cool.csv')

T = df['logT'].values
T = [10**x for x in T]
hfun0 = df['hfun0'].values
cfun0 = df['cfun0'].values


plt.scatter(np.log10(T), np.log10(cfun0), s = 10, color = 'blue')
plt.scatter(np.log10(T), np.log10(hfun0), s = 10, color = 'red')

plt.ylim(-26.4, -21.6)


plt.show()





