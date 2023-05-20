
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./Outputs/G-0.520030.csv')

x = data['x']
y = data['y']

print(x.shape)


xy = 0.25

plt.scatter(x, y, s = 0.01, color = 'k')

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.show()


