
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('GPU_IC_Anathpin_35k_RAND_from_65k.csv', names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'h', 'eps'])

plt.scatter(df['x'].values, df['y'].values, s = 0.002, color = 'k')

plt.show()



