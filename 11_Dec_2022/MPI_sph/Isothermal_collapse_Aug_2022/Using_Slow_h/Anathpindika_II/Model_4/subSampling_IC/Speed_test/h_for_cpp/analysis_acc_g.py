
import numpy as np
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt


dfgpu = pd.read_csv('acc_g_from_cpp_100k.csv')
ax_gpu = dfgpu['accx'].values
ay_gpu = dfgpu['accy'].values
az_gpu = dfgpu['accz'].values

dfcpu = pd.read_csv('acc_g.csv')
ax_cpu = dfcpu['accx'].values
ay_cpu = dfcpu['accx'].values
az_cpu = dfcpu['accx'].values


Y = ax_gpu - ax_cpu
X = np.arange(len(Y))

X = X[::100]
Y = Y[::100]

plt.figure(figsize = (15, 6))
plt.plot(X, Y, color = 'k')

#plt.ylim(-2.2, 2.2)


plt.show()







