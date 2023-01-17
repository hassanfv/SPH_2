import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('e_H_collision.csv') # Dalgarno and McCray 1972.

Temp = df['T'].values
Lambda_e_H = df['L'].values
Lambda_e_H = [float(x)* 1e-24 for x in Lambda_e_H]

plt.scatter(Temp, Lambda_e_H, s = 10)
plt.show()





