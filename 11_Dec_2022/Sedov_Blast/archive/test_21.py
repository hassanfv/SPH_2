
import numpy as np
import matplotlib.pyplot as plt
import pickle


stp = 1. / 24.

xL = np.arange(-.5, 0.0, stp)
xR = np.arange(0, 0.5, stp)
X = np.concatenate((xL, xR))

Y = X.copy()
Z = X.copy()

print(X)
print(len(X))


N = len(X)**3

r = np.zeros((N, 3))

print(r.shape)

i = 0

planeXY =[]

for x in X:

	for y in Y:

		for z in Z:
		
			r[i, 0] = x
			r[i, 1] = y
			r[i, 2] = z
			
			if z == 0.:
				planeXY.append(i) # saving the indices of the particles residing in the XY plane at Z = 0.0
			
			i += 1


X = r[:, 0]
Y = r[:, 1]
Z = r[:, 2]

nx = np.where(np.logical_and((X == 0.0), (Y == 0.0)))[0]
Z = Z[nx]

nz = np.where(Z == 0.)

print(nx[nz]) # the row 4210 corresponds to (x, y, z) = (0, 0, 0)


with open('SedovBlast_30.pkl', 'wb') as f:

	pickle.dump(r, f)


with open('PlaneXY.pkl', 'wb') as f:

	pickle.dump(planeXY, f)


plt.scatter(r[:, 0], r[:, 1], s = 1)

plt.scatter(X[nx[nz]], Y[nx[nz]], s = 50, color = 'r')

plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)

plt.show()






