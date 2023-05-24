
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from libsx import *


with open('Evrard_33552.pkl', 'rb') as f:
	data = pickle.load(f)


x = data['x'].reshape(-1, 1)
y = data['y'].reshape(-1, 1)
z = data['z'].reshape(-1, 1)

r = np.hstack((x, y, z))

N = len(x)

epsilon = np.zeros(N) + 0.05

vx = x * 0.0
vy = y * 0.0
vz = z * 0.0

v = np.hstack((vx, vy, vz))

print(v.shape)

uFloor = 0.05 #0.00245 # This is also the initial u.   NOTE to change this in 'do_sth' function too !!!!!!!!!!!!!!!!!!!!!
u = np.zeros(N) + uFloor # 0.0002405 is equivalent to T = 1e3 K

MSPH = 1.0 # total gas mass
m = np.zeros(N) + MSPH/N

Th1 = time.time()
#-------- h (initial) -------
h = do_smoothingX((r, r))  # This plays the role of the initial h so that the code can start !
#----------------------------
print('Th1 = ', time.time() - Th1)


x = np.round(r[:, 0], 5)
y = np.round(r[:, 1], 5)
z = np.round(r[:, 2], 5)

vx = np.round(v[:, 0], 5)
vy = np.round(v[:, 1], 5)
vz = np.round(v[:, 2], 5)

dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'h': np.round(h, 6), 'eps': epsilon, 'u': u}

df = pd.DataFrame(dictx)

num = str(int(np.floor(N/1000)))
df.to_csv('Evrard_GPU_IC_' + num + 'k.csv', index = False, header = False)

plt.scatter(x, y, s = 0.1, color = 'k')
plt.show()



