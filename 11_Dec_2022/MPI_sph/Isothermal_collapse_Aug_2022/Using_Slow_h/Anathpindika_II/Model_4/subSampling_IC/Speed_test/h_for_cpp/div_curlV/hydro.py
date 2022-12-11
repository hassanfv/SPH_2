
import numpy as np
import pickle
import pandas as pd


with open('hydroData_50k.pkl', 'rb') as f:
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


dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'rho': rho, 'P': P, 'c': c, 'h': h, 'm': m, 'divV': divV, 'curlV': curlV}
df = pd.DataFrame(dictx)

df.to_csv('Hydra_50k.csv', index = False)

print(df)

#x[i], y[i], z[i], vx[i], vy[i], vz[i], rho[i], P[i], c[i], h[i], m[i], divV[i], curlV[i]


#with open('Hydra_50k.txt', 'a') as f:

#	for i in range(len(x)):
#		f.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+' '+str(vx[i])+' '+str(vy[i])+' '+str(vz[i])+' '+str(rho[i])+' '+str(P[i])+' '+str(c[i])+' '+str(h[i])+' '+str(m[i])+' '+str(divV[i])+' '+str(curlV[i]))




