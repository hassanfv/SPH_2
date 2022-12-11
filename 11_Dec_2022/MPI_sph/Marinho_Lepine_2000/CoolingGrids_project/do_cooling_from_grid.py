
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

print(data.shape)

rho = data[:, 0]
uBefore = data[:, 1]
uAfter = data[:, 2]
dt = data[:, 3]

rhoA = rho.reshape(1000, 1000)

print(rhoA.shape)

s()


ut = 7.0
rhot = 2.0


#----- n11 = lower left -------
n11 = np.where(uBefore <= ut)[0]
u11 = np.max(uBefore[n11])
uc11 = np.max(uAfter[n11]) # u after cooling.


print(u11, uBefore[n11[-1]])
print(uc11, uAfter[n11[-1]])
s()


rhotmp = rho[n11]
n11t = np.where(rhotmp <= rhot)[0]
rho11 = np.max(rhotmp[n11t])
#------------------------------

#----- n12 = lower right ------
n12 = np.where(uBefore >= ut)[0]
u12 = np.min(uBefore[n12])
uc12 = np.min(uAfter[n12])

rho12 = rho11
#------------------------------

#----- n12 = upper left -------
u21 = u11
uc21 = uc11

n21 = np.where(rhotmp >= rhot)[0]
rho21 = np.min(rhotmp[n21])
#------------------------------

#----- n12 = upper right ------
u22 = u12
uc22 = uc12
rho22 = rho21
#------------------------------

n = [(u11, rho11, uc11), (u12, rho12, uc12), (u21, rho21, uc21), (u22, rho22, uc22)]

print(bilinear_interpolation(ut, rhot, n))
print(n)







