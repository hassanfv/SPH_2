
import numpy as np
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt


dfgpu = pd.read_csv('acc_g_from_cpp_double.csv')
ax_gpu = dfgpu['ax'].values
ay_gpu = dfgpu['ay'].values
az_gpu = dfgpu['az'].values

dfcpu = pd.read_csv('acc_g.csv')
ax_cpu = dfcpu['ax'].values
ay_cpu = dfcpu['ay'].values
az_cpu = dfcpu['az'].values


Y = ax_gpu - ax_cpu
X = np.arange(len(Y))

X = X[::100]
Y = Y[::100]

plt.figure(figsize = (15, 6))
plt.plot(X, Y, color = 'k')

#plt.ylim(-2.2, 2.2)


plt.show()







