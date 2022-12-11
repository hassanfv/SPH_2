
# The difference with _v3 is that here the particles can have different mass and the gravitational acceleration calculation is accordingly modified.
# The difference with _v2 is that here we incorporate shear viscosity by using the Balsara switch.
# The difference with previous version is that here we separated u and u_previous, ut_previous updates separately. See below.
# modified to be used with any number of CPUs.
# New h algorithm is employed !

import numpy as np
import time
import pickle
import os
from libsx2_2t import *
from shear_test3_t_del import *
import pandas as pd


np.random.seed(42)


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
#divV = data['divV']
#curlV = data['curlV']
alpha = data['alpha']



i = 1
j = 65

dx = x[i] - x[j]
dy = y[i] - y[j]
dz = z[i] - z[j]
rr = np.sqrt(dx*dx + dy*dy + dz*dz)
hij = 0.5 * (h[i] + h[j])
q = rr/hij


print('r_i = ', r[i, :])
print('r_j = ', r[j, :])
print('rr = ', rr)
print('q = ', q)


s()


divV, curlV = div_curlVel(r, v, rho, m, h)

dictx = {'divV': divV, 'curlV': curlV}

df = pd.DataFrame(dictx)

df.to_csv('div_curlV.csv', index = False)

#print(df)





