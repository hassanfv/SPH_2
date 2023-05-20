
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob


# 'pos': r, 'v': v, 'm': m, 'u': u, 'dt': dt, 'current_t': t, 'rho': rho, 'NG': NG, 'ND': ND

filz = np.sort(glob.glob('./Outputs/*.pkl'))

print(filz)

j = -1

with open(filz[j], 'rb') as f:

	data = pickle.load(f)

# distances = np.linalg.norm(gas_positions - bh_position, axis=1)

r = data['pos']
NG= data['NG']
ND= data['ND']
r = r[:NG]
u = data['u']

x = r[:, 0]
y = r[:, 1]
z = r[:, 2]

dd = (x*x + y*y + z*z)**0.5
nd = np.argsort(dd)[:64] # we need particles with smallest distances
print('d = ', dd[nd])
print('dddd = ', np.sort(dd)[:20])

nx = np.argsort(u)[-64:]
xt = r[nx, 0]
yt = r[nx, 1]
zt = r[nx, 2]

print()
print('u = ', u[nx])
print()
print(filz[j])
print('NGas = ', NG)
print('NDM = ', ND)


xy = 0.1

plt.scatter(r[:NG, 0], r[:NG, 1], s = 0.02, color = 'k')
plt.scatter(xt, yt, s = 10, color = 'red')
plt.scatter(x[nd], y[nd], s = 4, color = 'lime')
plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.show()

