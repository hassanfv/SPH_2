
import time
import numpy as np
import random
from libsx2_2t import *
import pickle
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

NSample = 212472 # The desired number of particles to be included in ONE cloud in the IC.
NSample = int(NSample)

with open('Main_IC_Grid_113k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']

print(r.shape)

if NSample > r.shape[0]:
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('WARNING !!! NSample is greater than r.shape !!! Please change NSample!!!!')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	exit(0)

Ntot = r.shape[0]
nn = np.arange(Ntot)
np.random.shuffle(nn)
rows = nn[:NSample]

r = data['r'][rows, :]
v = data['v'][rows, :]
m = data['m'][rows]

m = m / np.sum(m) # Normalizing m! VERY IMPORTANT!!

h = data['h'][rows]
rho = data['rho'][rows]
u = data['u'][rows]

paramz = data['paramz']
paramz[0] = NSample # only NSample meeded to be updated!

#***********************************************
#**************** FOR GPU **********************
#***********************************************

x = np.round(r[:, 0], 5)
y = np.round(r[:, 1], 5)
z = np.round(r[:, 2], 5)

vx = np.round(v[:, 0], 5)
vy = np.round(v[:, 1], 5)
vz = np.round(v[:, 2], 5)

epsilon = 0.1 + np.zeros(len(x))

dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'h': np.round(h, 6), 'eps': epsilon, 'u': u}

df = pd.DataFrame(dictx)

num = str(int(np.floor(NSample/1000)))
df.to_csv('GPU_IC_Arrega_' + num + 'k.csv', index = False, header = False)

#-------
try:
	os.remove('params.GPU')
except:
	pass

outF = open('params.GPU', "a")
for i in range(len(paramz)):
	outF.write(str(paramz[i]))
	outF.write('\n')
outF.close()
#************** END of GPU IC Creation *********

print('m = ', np.sort(m))

print()
print(r.shape)
print()
print('Done !')

plt.scatter(x, y, s = 0.5, color = 'k')
plt.show()






