
import pickle
import pandas as pd
import numpy as np


with open('coolingGrid_J_0.0.pkl', 'rb') as f:
	data = pickle.load(f)


cols = ['u_ad', 'rho', 'dt', 'delta_u'] # all in physical cgs unit. (delta_u = u_ad - u_after_cooling)
					# Note again: u_after_colling = u_ad + delta_u !!!!!!!

df = pd.DataFrame(data, columns = cols)
df = df.sort_values(['u_ad', 'rho'], ascending = [True, True])

df['u_ad'] = df['u_ad'].apply(lambda x: '%.4E' % x)
df['rho'] = df['rho'].apply(lambda x: '%.4E' % x)
df['dt'] = df['dt'].apply(lambda x: '%.4E' % x)
df['delta_u'] = df['delta_u'].apply(lambda x: '%.4E' % x)

df.to_csv('coolingGrid_J_0.0.csv', index = False)

print(df)
