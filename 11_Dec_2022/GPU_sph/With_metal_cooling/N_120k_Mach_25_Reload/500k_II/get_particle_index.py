import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import readchar
import time
import os


filz = np.sort(glob.glob('./Outputs/*.csv'))

j = -1

df = pd.read_csv(filz[j])
	
x = df['x'].values
y = df['y'].values
z = df['z'].values

#plt.scatter(y, z, s = 0.01)
#plt.show()

nn = np.where((z > 0.09) & (z < 0.11) & (y < -0.18) & (y > -0.2))[0]

print(nn)
