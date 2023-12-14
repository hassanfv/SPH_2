
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')

x = df['x'].values
y = df['y'].values
z = df['z'].values

h = df['h'].values


plt.hist(h, bins = np.linspace(0, 0.019, 20))
plt.show()


plt.scatter(x, y, s = 0.01)
plt.scatter([0], [0], s = 10, color = 'r')
plt.show()




