
import time
import numpy as np
import random
from libsx2_2t import *
import pickle
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

NSample = 2.0 * 90000 #113104 #212472 # The desired number of particles to be included in ONE cloud in the IC.
NSample = int(NSample)

with open('Main_IC_Grid_90k.pkl', 'rb') as f:
	data = pickle.load(f)

r = data['r']

print(r.shape)


if NSample/2 > r.shape[0]:
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('WARNING !!! NSample is greater than r.shape !!! Please change NSample!!!!')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	exit(0)

Ntot = r.shape[0]
nn = np.arange(Ntot)
np.random.shuffle(nn)
rows = nn[:NSample]

rL = data['r'][rows, :]
rR = rL.copy()
rL[:, 0] = rL[:, 0] - 1.0
rR[:, 0] = rR[:, 0] + 1.0 # The two cloud will be touching in the begining to save time.
r = np.vstack((rL, rR))

m = data['m'][rows]
m = m / np.sum(m) # Normalizing m! VERY IMPORTANT!!
m = np.hstack((m, m))

h = data['h'][rows]
h = np.hstack((h, h))

rho = data['rho'][rows]
rho = np.hstack((rho, rho))

u = data['u'][rows]
u = np.hstack((u, u))

paramz = data['paramz']
paramz[0] = NSample # only NSample meeded to be updated!

#----- VELOCITY Section ------
unitMass_in_g = paramz[5]
unitLength_in_cm = paramz[4]
grav_const_in_cgs = paramz[8]
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
print(f'unitTime_in_s = {unitTime_in_s:.4E}')
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s
c_s = paramz[1]
Mach = paramz[7]
vL = np.zeros_like(rL)
vR = vL.copy()
vL[:, 0] = 0.5 * Mach * c_s / unitVelocity_in_cm_per_s
vL = np.round(vL, 4)
vR[:, 0] = -0.5 * Mach * c_s / unitVelocity_in_cm_per_s
vR = np.round(vR, 4)
v = np.vstack((vL, vR))

#***********************************************
#**************** FOR GPU **********************
#***********************************************

x = np.round(r[:, 0], 5)
y = np.round(r[:, 1], 5)
z = np.round(r[:, 2], 5)

vx = np.round(v[:, 0], 5)
vy = np.round(v[:, 1], 5)
vz = np.round(v[:, 2], 5)

epsilon = 0.05 + np.zeros(len(x))

dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'h': np.round(h, 6), 'eps': epsilon, 'u': u}

df = pd.DataFrame(dictx)

num = str(int(np.floor(NSample/1000)))
df.to_csv('GPU_IC_DLA_' + num + 'k.csv', index = False, header = False)

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
print('sum(m) (we have 2 clouds !!!) = ', np.sum(m))

print()
print(r.shape)
print()
print('Done !')

plt.figure(figsize = (12, 6))
plt.scatter(x, y, s = 0.5, color = 'k')
plt.show()






