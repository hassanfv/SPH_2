
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from libsx import *


with open('Evrard_65752.pkl', 'rb') as f:
	data = pickle.load(f)


x = data['x'].reshape(-1, 1)
y = data['y'].reshape(-1, 1)
z = data['z'].reshape(-1, 1)

r = np.hstack((x, y, z))

N = len(x)

epsilon = np.zeros(N) + 0.10

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

Typ = np.full(len(x), 0)

Typ = Typ.astype(np.int32)

x = x.astype(np.float32)
y = y.astype(np.float32)
z = z.astype(np.float32)

vx = vx.astype(np.float32)
vy = vy.astype(np.float32)
vz = vz.astype(np.float32)

m = m.astype(np.float32)
h = h.astype(np.float32)
epsilon = epsilon.astype(np.float32)
u = u.astype(np.float32)

# Save the arrays to a binary file
num = str(int(np.floor(N/1000)))
with open('Evrard_GPU_IC_' + num + 'k.bin', "wb") as file:
    file.write(Typ.tobytes())
    file.write(x.tobytes())
    file.write(y.tobytes())
    file.write(z.tobytes())
    
    file.write(vx.tobytes())
    file.write(vy.tobytes())
    file.write(vz.tobytes())
    
    file.write(m.tobytes())
    file.write(h.tobytes())
    file.write(epsilon.tobytes())
    file.write(u.tobytes())

plt.scatter(x, y, s = 0.1, color = 'k')
plt.show()




