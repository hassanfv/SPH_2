
import numpy as np
import matplotlib.pyplot as plt
import pickle


N = 34

xL = np.linspace(-0.40, 0, int(N/2))
xR = np.linspace(0, 0.40, int(N/2))


X = np.concatenate((xL[:-1], xR)) # [:-1] is used to exclude one of the zeros !

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


with open('SedovBlast.pkl', 'wb') as f:

	pickle.dump(r, f)


with open('PlaneXY.pkl', 'wb') as f:

	pickle.dump(planeXY, f)


#plt.figure(figsize = (5, 5))

plt.scatter(r[:, 0], r[:, 1], s = 1)

plt.scatter(X[nx[nz]], Y[nx[nz]], s = 50, color = 'r')

plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)

plt.show()






