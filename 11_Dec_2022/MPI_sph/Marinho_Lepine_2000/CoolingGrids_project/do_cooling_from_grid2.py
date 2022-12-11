
import numpy as np
import pickle


def bilinear_interpolation(x, y, points):
    # Source: https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


with open('u_after_cool_MPI_dt_0.005.pkl', 'rb') as f:
	data = pickle.load(f)


NN = int(np.sqrt(data.shape[0]))

rho = data[:, 0].reshape(NN, NN)
uBefore = data[:, 1].reshape(NN, NN)
uAfter = data[:, 2].reshape(NN, NN)
dt = data[:, 3].reshape(NN, NN)

u_uniq = np.unique(uBefore)
rho_uniq = np.unique(rho)

print('u range = ', np.min(u_uniq), np.max(u_uniq))
print('rho range = ', np.min(rho_uniq), np.max(rho_uniq))

#print(rho.shape, uBefore.shape, uAfter.shape)


ut = 0.2
rhot = 20.


#-------- n11 i.e. lower left -----------
n11 = np.where(u_uniq <= ut)[0]
u11 = np.max(u_uniq[n11])

n11 = np.where(rho_uniq <= rhot)[0]
rho11 = np.max(rho_uniq[n11])
#----------------------------------------

ndx = np.where((uBefore == u11) & (rho == rho11))

nrow = ndx[0][0]
ncol = ndx[1][0]

u11Z = uAfter[nrow, ncol]
print('u11, rho11, u11Z = ', u11, rho11, u11Z)

##-------- n12 i.e. lower right ---------
u12 = uBefore[nrow+1, ncol] # nrow movies in the u direction. ncol moves in the rho direction
rho12 = rho[nrow+1, ncol]
u12Z = uAfter[nrow+1, ncol]

print('u12, rho12, u12Z = ', u12, rho12, u12Z)
#----------------------------------------

##-------- n21 i.e. upper left ----------
u21 = uBefore[nrow, ncol+1]
rho21 = rho[nrow, ncol+1]
u21Z = uAfter[nrow, ncol+1]

print('u21, rho21, u21Z = ', u21, rho21, u21Z)
#----------------------------------------

##-------- n22 i.e. upper right ---------
u22 = uBefore[nrow+1, ncol+1]
rho22 = rho[nrow+1, ncol+1]
u22Z = uAfter[nrow+1, ncol+1]

print('u22, rho22, u22Z = ', u22, rho22, u22Z)
#----------------------------------------



s()



n = [(u11, rho11, uc11), (u12, rho12, uc12), (u21, rho21, uc21), (u22, rho22, uc22)]

print(bilinear_interpolation(ut, rhot, n))
print(n)







