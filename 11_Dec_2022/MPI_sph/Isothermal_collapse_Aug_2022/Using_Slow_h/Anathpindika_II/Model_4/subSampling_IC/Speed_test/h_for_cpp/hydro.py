
import numpy as np
import pickle
import pandas as pd


with open('hydroData.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']
x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

v = data['v']
vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

rho = data['rho']
P = data['P']
c = data['c']
h = data['h']
m = data['m']
divV = data['divV']
curlV = data['curlV']
alpha = data['alpha']

print(r.shape)

dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'rho': rho, 'P': P, 'c': c, 'h': h, 'm': m, 'divV': divV, 'curlV': curlV}
df = pd.DataFrame(dictx)

df.to_csv('Hydra_130k.csv', index = False)

print(df)




